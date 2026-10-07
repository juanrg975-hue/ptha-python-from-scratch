# from_scratch_v12

> **This is from_scratch_v12 (2026-10-06).** A copy of `from_scratch_v11/`
> (packages `pyptha_v12`, `python_logic_tree_v12`) that adds
> **`--mesh-file`: an external quadrilateral mesh as the geometry** instead of
> SLAB (steps 1-2), see [What v12 changes](#what-v12-changes), and (2026-10-07)
> **`--cell-size strasser`: no unit source larger than the smallest rupture /
> k**, see [`--cell-size strasser`](#--cell-size-strasser-2026-10-07). Without
> them (defaults: no mesh file, `--cell-size mean`) v12 is v11. v11 = v10_q plus PTHA18's
> **variable shear modulus** rates (step 7c). Every file steps 1-7b write is
> the same as v10_q's (checked byte for byte on calabria2); step 7c only adds
> files. v10_q itself is a copy of v10, itself a copy of `from_scratch_v9/`.
> With the defaults it gives v9's results except where the trench ramp rule
> changes the trench (hellenic, caribbean/antilles, kermadec); everything else
> new is opt-in or additive. What each version adds:
>
> | version | adds | option | default | section |
> |---|---|---|---|---|
> | v10 | each uniform-slip rupture sized from the real km of ITS cells (one cell count per magnitude in rptha) | `--rupture-size rptha\|local` | `rptha` (= v9) | [What v10 changes](#what-v10-changes) |
> | v10_q | with `local`, PTHA18's conditional-probability weight times q, a rupture-overlap correction | (part of `--rupture-size local`) | off | [What v10_q changes](#what-v10_q-changes) |
> | v10_q | trench ramp rule: a trench end that climbs in depth like a ramp is the edge of the SLAB data and is trimmed before the datum is built | `--trench-ramp on\|off` | `on` | [Trench ramp rule](#trench-ramp-rule-v10_q) |
> | v10_q | HS and VAUS rates as PTHA18 computes them (peak-slip limit + DART weights), in step 7b | none | on | [step 7b](#step7b_stochastic_slippy-heterogeneous-hs-and-variable-area-uniform-vaus-slip-variants-optional) |
> | v11 | variable shear modulus: PTHA18's `variable_mu` rates (FAUS, HS, VAUS, curves, percentiles), new step 7c | `--variable-mu on\|off` | `off` (constant 30 GPa) | [What v11 changes](#what-v11-changes) |
> | v11 | step 8 (official PTHA18 run) works from `V9/`: PTHA18's extracted files are looked for one level up too | none | | [step 8](#step8_officialpy-official-ptha18-run-optional) |
> | v12 | an external quadrilateral mesh as the geometry instead of SLAB | `--mesh-file FILE` | none (SLAB) | [What v12 changes](#what-v12-changes) |
> | v12 | unit sources capped by the scaling relation: no cell larger than the Mmin rupture / k | `--cell-size mean\|strasser`, `--cell-k K` | `mean` (= v11), k 1.5 | [`--cell-size strasser`](#--cell-size-strasser-2026-10-07) |
>
> Why, with every figure and number: `V9/html/v10q_rationale.html` (the case
> for v10_q on Calabria, Caribbean and two control zones) and
> `V9/html/slip_comparison_v9_v10.html` (slip and sea-floor uplift, Calabria).
> Known limits still open: [Known limitations (v11)](#known-limitations-v11),
> [(v10_q)](#known-limitations-v10_q).
> Example folders (v11, `--rupture-size local`, constant rigidity): `calabria2_v11`,
> `caribbean2_v11`, `kermadectonga2_v11`, `kurilsjapan_v11`; with `--variable-mu on`:
> `calabria2_v11_varmu`. Older: `calabria2_v10`, `calabria2_v10_q`,
> `caribbean2_v10`, `caribbean2_v10_q`, `hellenic_west2_v10_q`
> ([recipes](#recipes-v10-and-v10_q)).
> The documentation pages are in [html/docs/index.html](html/docs/index.html)
> ("What is new in v11", "What is new in v10 and v10_q").

A generator that builds a complete, self-contained **from-scratch PTHA18
source-zone example** (geometry, convergence, seismicity, logic tree, rates,
HTML report) for any subduction zone covered by SLAB2.0 / SLAB1.0, using
**public data only**.

v9 is a copy of `from_scratch_v8/` that adds **LEVEL 0 segmentation**, the one
part of the PTHA18 logic tree v8 did not reproduce. `from_scratch_v8/` itself
was not touched, and neither was anything it produced. v9's engine gives
v8's results bit for bit from the same input (`validation/validate_v9.py
v8parity`, all 32,000 logic-tree branches and 3,075 scenario rates), but
since 2026-09-30 step 6 feeds it two different zone values from Berryman
(coupling and mw_max_observed, see [What v9 adds](#what-v9-adds)), so an
unsegmented v9 run is no longer identical to v8's on the zones they change.

v8 was itself a copy of `from_scratch_v7/` after a full audit against rptha (the
R package PTHA18 was built with) and PTHA18's own published files.

**Kept in step with v8.** Every v8 change made after v9 was forked is in v9
too (synced 2026-09-30, see [7e](#7e-what-v8-changed)): v8.1's plate-boundary
check and `--auto-clip`, PTHA18's own cell outlines in the official mesh
figures, the always-on depth profile and the `outputs/exports/*.csv` files;
v8.2's `--columns trench|average` and the mesh-shape printout. On a
segmented run the exports carry the segments too: `mesh_cells.csv` and
`convergence_profile.csv` get a `segment` column (the segment each
along-strike column belongs to), and `posterior_mean_rate_curve.csv` has one
column per segment, the union and the zone's LEVEL 0 mix, as `rate_curves.csv`.

> **The complete documentation**, one page per topic with figures, worked
> examples and diagrams, is [html/docs/index.html](html/docs/index.html).

### What v12 changes

**Why.** Colleagues asked whether ruptures and rates could be generated from
a mesh they already have: a text file with one line per subfault and its four
corners as lon, lat, depth (`alaskaaleutians_quadrilateral_coors.dat`, 312
quadrilaterals). v12 adds that as an input.

**How.** `generate.py --mesh-file FILE [--mesh-depth-units auto|m|km]` copies
the file into the example's `inputs/geometry/` and:
- step 1 does not use SLAB (nothing to fetch);
- step 2 reads the mesh with `pyptha_v12/mesh_file.py` instead of meshing
  contours, and saves the usual node array;
- step 6 writes `"geometry": {"mode": "mesh_file", "mesh_file": ...}`, which
  the engine (`run_logic_tree.build_grid`) and step 7b read;
- steps 3-9 run unchanged (convergence, GCMT, logic tree, ruptures, rates,
  HS/VAUS, step 8's official comparison, the report).

`mesh_file.read_quadrilateral_mesh` needs a STRUCTURED mesh (rows down dip x
columns along strike, cells sharing their corners) and finds everything else
itself: the order of the lines and of the corners in each line does not
matter; depth units (metres if any |depth| > 200) and sign are detected; each
cell's top, bottom and sides come from the corners it shares with its
neighbours, and depth decides once, for the whole mesh, which way is down dip
(2026-10-06: until then each cell's shallowest edge was its top, which broke
the rows of fine meshes on flat or skewed slabs, e.g. alaskaaleutians,
hellenic2 and hellenic_west2 at ~36 x 29 km); rows and columns come from the
corners the cells share; the columns are reversed if needed so the fault dips
to the right of the strike (rptha's convention). Anything else (triangles,
holes, ragged columns) is refused with a message.

**Checked** (`validation/validate_v12_mesh.py`, `tests/test_mesh_file.py`):
- the colleagues' file IS PTHA18's alaskaaleutians mesh: 78 x 4 = 312 cells,
  centres, strikes and the cells of rows 2-4 equal to PTHA18's unit-source
  table to 1e-10; only the trench row differs, because the file puts the
  trench at 0.1 km (-100 m) where PTHA18 has 0 km (centre depth +0.05 km, dip
  -0.19 deg, area -0.008%);
- the engine on PTHA18's official alaskaaleutians input with the file in
  place of PTHA18's table: with the trench moved to 0 km, the same 4,111
  scenarios and rates to 8.5e-12 (so the mesh path is exact); with the file as
  sent, rates within 0.07% (Mw >= 7.2: x0.99996);
- 8 tests: the same synthetic mesh written with shuffled lines, rotated or
  reversed corner rings, km or m, either depth sign, or dipping the wrong way,
  always read back as the same node array; bad files refused. 187 passed,
  15 skipped. A 9th (2026-10-06): cells that deepen more across strike than
  down dip keep their rows.

**Example:** `alaskaaleutians_v12` (`generate.py alaska --zone alaskaaleutians
--folder alaskaaleutians_v12 --ptha false --rupture-size local --mesh-file
inputs_meshes/alaskaaleutians_quadrilateral_coors.dat`): steps 1-9 complete, step 8 against
PTHA18's official run; report.html has a "STEP 1-2: geometry from an external
mesh" card.

#### `--cell-size strasser` (2026-10-07)

**Why.** rptha's (and PTHA18's) rule sizes the unit sources ON AVERAGE: the
columns from the trench length / 50 km, the rows from the MEAN down-dip
length / 50 km. Where the zone is a fan (the trench much longer than the deep
edge: calabria2 981 km against 243 km) or its width changes along strike
(antilles2, alaskaaleutians, kermadectonga2), some cells come out far larger
than that: calabria2's trench cells reach 4042 km2, more than a whole Mw 7.2
rupture (Strasser: 2390 km2), its deep cells 72 km wide, alaskaaleutians' and
antilles2's ~100 km. A rupture cannot be smaller than one cell, so there the
local-size ruptures and, above all, the HS/VAUS footprints cannot take the
size and shape the scaling relation asks for.

**The rule.** `generate.py ... --cell-size strasser [--cell-k K]`: NO unit
source longer or wider than the rupture the scaling relation gives at the
event table's smallest magnitude, divided by K:

    L <= L_Strasser(Mmin) / K,   W <= W_Strasser(Mmin) / K   (every cell)

With Mmin = 7.2 (step 6) Strasser's rupture is 54.3 x 44.2 km, so K = 1.5
(default) caps the cells at 36.2 x 29.5 km and K = 2 at 27.2 x 22.1 km.
K = 1.5 means: no cell larger than an Mw 7.2 rupture one sigma smaller than
Strasser's mean in length and width (Strasser's log10 sigmas 0.180 and 0.173
are factors 1.51 and 1.49).

**How.** Same contours, same `optimal` mesher, same rows x columns structure:
`contour_discretisation.discretized_source_from_contours_bounded` starts from
rptha's rule with the caps as targets, then raises the row count until the
widest cell (v12's own unit-source width) meets the W cap and the column
count until the longest cell meets the L cap (2% tolerance), re-meshing each
time. `--columns`, the 50 km targets and any fixed row count (`--ptha true`,
`N_DOWNDIP_OVERRIDE`) are not used. Step 2 holds `CELL_SIZE`, `CELL_K`,
`CELL_CAP_MW` (7.2) and `CELL_CAP_RELATION` ("Strasser"); step 6 writes
`cell_size`, `cell_k`, `cell_cap_mw` and `cell_cap_relation` into the
geometry block, and they enter the mesh fingerprint, so steps 7, 7b and 9
reuse step 2's mesh (or rebuild the same one). With `mean` (default) none of
these fields is written and every file is what it was. Needs
`--discretizer optimal`; cannot be combined with `--mesh-file`. Meant for
`--rupture-size local`: the cells are still unequal (on an arc they cannot be
equal), so rptha's one-block-per-magnitude rule does not use them well
(calabria2 at K = 1.5: 31% of ruptures within a factor 1.5 of Strasser's area
and shape with `rptha`, 96% with `local`).

**What it changes and costs** (10 zones, `examples_newtest/`, report in
`examples_newtest/html/report_strasser_meshes.html`):
- rates per magnitude: unchanged (new / old exceedance rate 1.00-1.06 at
  Mw 7.5-9.0; total area and dip are kept);
- HS/VAUS: with ~50 km cells a footprint cannot be smaller than one cell, so
  at Mw 7.2-7.5 the compact, high-slip realisations the area variability is
  meant to produce are largely missing (kurilsjapan Mw 7.2: 70% of VAUS are a
  single cell, VAUS area p05 0.88 x Strasser instead of ~0.35; calabria2
  Mw 7.2: median VAUS area 1.25 x Strasser). With K = 1.5 the VAUS area
  distribution is the same at every magnitude (p05 ~0.35, median ~1.0);
- uniform-slip ruptures within a factor 1.5 of Strasser's area AND
  length/width (Mw 7.2-8.4): 49-68% -> 87-99%. Their areas were already
  right; the gain is in the shape, mostly within Strasser's own scatter;
- cost: 2.9-7.3 times more unit sources (each one a tsunami simulation),
  1.8-5.2 times more ruptures, pipeline up to ~13 min (K = 1.5) and ~34 min
  (K = 2) on the longest zones; scenarios no longer match PTHA18's one by one.
  K = 2 barely improves on K = 1.5 and doubles the cells again.

The gain is largest where the usual mesh fails by structure (fans: calabria2,
hellenic; zones of varying width: antilles2, alaskaaleutians,
kermadectonga2) and smaller in regular zones (makran2, kurilsjapan,
puysegur2), where any finer mesh would give it. Its effect on the tsunami
hazard itself has not been measured.

**Checked:** `tests/test_cell_size.py` (a synthetic fan: the `mean` mesh
breaks the cap, the bounded one meets it, no defects, area within 3%); the
function reproduces the meshes of `examples_newtest/k1_5` and `k2`
(built before the flag existed) node for node on calabria2, hellenic_west2,
puysegur2 and makran2; end-to-end on calabria2 (`examples_newtest/flag_check/`,
steps 1-9 without 8): without the flag every output (mesh, rate curves,
percentiles, scenario rates, HS and VAUS fields) is identical to
`examples/calabria2`'s, so `mean` is v11 bit for bit; with
`--cell-size strasser --cell-k 1.5` steps 7, 7b and 9 reuse step 2's mesh
(fingerprint matches) and every output equals the `--mesh-file` run of the
same mesh to 1e-10 (that run's file rounded the nodes to 1e-10 deg). Invalid
combinations (`--mesh-file`, `--discretizer lm|mid`) are refused before
anything is written. 184 tests pass (2 new).

**Example:** `generate.py calabria --zone calabria2 --folder calabria2_k15
--ptha false --rupture-size local --cell-size strasser --cell-k 1.5`:
30 x 13 = 390 unit sources (100 with `mean`).

### What v11 changes

**Why.** Rigidity (shear modulus, mu) is how stiff the rock is. Steps 7 and
7b use 30 GPa everywhere, PTHA18's `rate_annual`. Real rock is softer near
the trench (about 10 GPa) and stiffer at depth (up to 67 GPa), and for the
same earthquake as a seismometer measures it (the same seismic moment,
M0 = mu x area x slip) soft rock needs more slip: a shallow rupture makes a
bigger tsunami than its constant-rigidity magnitude says. PTHA18 publishes a
second set of rates for this, `variable_mu_rate_annual`, and its headline
hazard maps use them with the HS scenarios (PTHA18 report Sections 3.7.5 and
4.1). The tsunami maps these outputs are for need them.

**How PTHA18 does it (ported in step 7c).**
1. Each cell's rigidity from its depth: rptha `shear_modulus_depth`, a fit
   to Bilek & Lay (1999): 10 GPa down to 7.5 km, 30 GPa at 15 km, 67 GPa from
   35 km, log-linear between (`pyptha_v12/variable_mu.py`).
2. Every scenario keeps its cells and its slip; only its magnitude is
   relabelled: M0 = sum(area x slip x mu(depth)) (`variable_mu_Mw`).
3. The rates stay functions of the constant-rigidity magnitude, but the GCMT
   earthquakes of LEVEL 3 have real magnitudes. The difference, measured on
   the run's own HS scenarios (`make_conditional_ecdf`, given the magnitude),
   is treated as an observation error of the catalogue when the branch
   weights are updated: a second set of weights,
   `posterior_prob_with_Mw_error` (v9's engine already computed it with an
   error model; nothing fed it until v11).
4. The variable rates are the same branch curves with those weights
   (`MwRateFunction.with_mw_error`): scenario rates and 5 percentiles, rate
   curves, LEVEL 5, the zone mix on segmented runs (each segment uses the HS
   scenarios touching it, as rptha).
5. HS and VAUS share the variable FAUS rate with PTHA18's DART curves for that
   case, a second pair recovered from its published rates like the constant
   ones (`data/ptha18_peak_slip_quantile_weights.csv` now has 4 columns).

**What does not change.** No scenario: same cells, slip, sea-floor
deformation and tsunami. Only how often each happens, through the LEVEL 3
weights. calabria2 (no GCMT earthquake): on the constant-magnitude scale the
rates move by +4% (Mw 7.5) to -6% (Mw 9.0), TVD between the two posteriors
0.037; counted by their real magnitude they are 25-35% lower (65% at Mw 9.0),
because most of the rate sits on the shallow rows, whose real Mw is 0.32
lower. A real Mw 8 on calabria2's shallow rows is a constant-rigidity Mw 8.3,
with that slip.

**Checked against PTHA18** (`validation/validate_v11.py`): every published
`variable_mu_Mw` of puysegur2 and kermadectonga2 (FAUS, HS, VAUS, 101,000+
scenarios) to 4e-15; the engine on PTHA18's puysegur2 input with PTHA18's
own HS catalogue gives `variable_mu_rate_annual` and its 5 percentiles to
7e-13, zero rates in place; every published HS and VAUS rate of both zones,
both rigidities, to 4e-12. The 4 DART curves pass the report's Section 3.6
method (inverse of a cubic through (0,0) and (1,1), residual < 5e-4).
Tests: 179 passed, 15 skipped (`test_variable_mu.py`, 7).

**Outputs** (all in `outputs/`, nothing else changes):
`scenario_rates_<zone>_variable_mu.csv` (FAUS: `Mw`, `variable_mu_Mw`, rates),
`rate_curves_variable_mu.csv`, `exceedance_rate_percentiles_variable_mu.csv`,
`variable_mu_deviation_<zone>.csv`, `logic_tree_branches_<zone>_variable_mu.csv`,
`logic_tree_summary_variable_mu.csv`, and new columns `variable_mu_Mw`,
`variable_mu_weight_in_family`, `variable_mu_rate_*` in both HS/VAUS
`summary.csv`. `report.html` gets a STEP 7c card, and the rupture explorer
gives every block its variable-rigidity magnitude. Page:
[html/docs/v11.html](html/docs/v11.html).

**Switch.** `generate.py --variable-mu on` (default `off`: constant
rigidity only, as all comparisons with PTHA18's `rate_annual`). Example:
`calabria2_v11_varmu`.

**Engine input.** A top-level block
`"variable_shear_modulus": {"hs_events": "<HS summary.csv or PTHA18 nc>", "curve": "default"|"prem"}`
switches it on (step 7c writes `inputs/input_<zone>_scratch_variable_mu.json`);
without it the engine is v10_q's.

### Known limitations (v11)

- **Only PTHA18's rigidity curve** (or PREM). A Mediterranean profile would be
  a new curve, not PTHA18's.
- **The depth is PTHA18's**: the curve is applied to the unit sources' `depth`,
  which is measured below the nearby trench (PTHA18's convention), while
  Bilek & Lay's depths include the water. PTHA18 does exactly this (its
  published `variable_mu_Mw` is reproduced to 4e-15), so v11 does too.
- **The DART curves** (HS/VAUS weights) come from 18 Pacific and Indian Ocean
  tsunamis (2006-2016); there are no DART buoys in the Mediterranean. PTHA18
  applies them to every zone, since they correct the slip generator, not a
  place; for Calabria that is the same assumption.
- **The segmented official check** (kermadectonga2) was not run for the
  variable rates: same code path as the constant one (validated to 1e-10),
  with the variable weights.
- Step 7c needs step 7b's HS scenarios (`--skip-hs` skips both), and runs
  only in examples generated with `--variable-mu on` (default `off`; or run
  `step7c_variable_mu.py --force` by hand).

### Trench ramp rule (v10_q)

**Why.** Step 1 measures every contour below the NEARBY trench (PTHA18's
convention), so a wrong stretch of trench becomes a wrong datum for
everything near it. At hellenic's Kefalonia end the raster's up-dip edge is
not trench but the edge of the SLAB2 data: 19.7 km below sea level, climbing
to the trench's normal 11.5 km over about 150 km. It was the datum of the
northern half of hellenic_west2, so on a slab that is almost flat there its
25-35 km contours piled into one line, and the mesh had a crease of sheared,
twisted cells (13 km long cells, corners 81 deg off square, twist 0.44).

**The rule** (`lib/slab_contours.find_trench(ramp_rise_km=)`, `_ramp_trim`):
each end of the trench is trimmed back, by at most 100 km, while its 50 km
running-median depth is more than 3 km deeper than the trench's normal depth
there (the larger of its median 100-200 km in from that end and the whole
trench's median). `generate.py --trench-ramp on` (default) writes
`TRENCH_RAMP_RISE_KM = 3.0` in step 1; `off` writes `None`, v9's trench.
Folders generated before this option have no such constant and keep v9's
trench even if re-run.

**Which zones change** (checked on every raster the V9 folders use): hellenic
(71 km at the Kefalonia end; hellenic_west2, hellenic2, and slightly
hellenic_east2, which shares the raster and datum), caribbean / antilles2
(68 km, a deep ramp at the start of the trench: antilles2's trench p95 depth
14.5 -> 11.2 km) and kermadectonga2 (11 km). Calabria, Kurils-Japan, Makran,
Puysegur and South America are identical.

**hellenic_west2** (`hellenic_west2_v10_q`, with `--rupture-size local`):
20 x 8 -> 19 x 8 cells; smallest cell 13 x 35 -> 19 x 34 km; worst corner
81 -> 52 deg off square (p90 50 -> 31); twist 0.44 -> 0.07; area 253,725 ->
234,639 km2 (-7.5%, the trimmed stretch was never trench), so the rates
change: Mw >= 7.2 -1.2%, >= 8.0 -2.3%, >= 8.8 -5.6%. The crease is gone. Tests:
`pyptha_v12/tests/test_trench_ramp.py`.

### What v10_q changes

**Why.** `--rupture-size local` makes one rupture per starting cell, so a part
of the zone with small cells (calabria2: the deep rows) is covered by more
ruptures per km2. PTHA18 splits each magnitude's rate among its ruptures in
proportion to area x convergence; with rptha's fixed cell count the area
itself is proportional to the cell size, which compensated that by accident.
With `local` the areas are all close to Strasser's, the compensation is
gone, and in calabria2_v10 the rate of Mw <= 7.6 moved from the trench row
(30% -> 18%) to the deep rows.

**q** (`pyptha_v12/events.py`, `coverage_weights`): for each rupture, the
mean over its cells of 1 / (number of ruptures of the same magnitude that
contain that cell). The engine multiplies the weight of whichever
conditional-probability model the input uses by q, only with
`--rupture-size local`. A part covered by three times as many ruptures gives
each of them a third of the weight, so the rate follows area and
convergence again. q is also larger on the zone's edges (fewer ruptures
cover them), so it does part of the job of PTHA18's edge correction, which
is still fitted afterwards (calabria2: multiplier 3.69 in v9, 1.66 in v10).

The total rate of every magnitude, and so every exceedance-rate table, is
unchanged (it comes from the moment balance, before the rate is split
among ruptures); what changes is each rupture's rate (`scenario_rates_*`,
`conditional_prob_*`) and the integrated slip. Tests: three more in
`pyptha_v12/tests/test_rupture_size.py`.

### What v10 changes

**The problem.** rptha builds the uniform-slip (FAUS) ruptures of a
magnitude as ONE block of cells (for example 2 along strike x 1 down dip at
Mw 7.2), sized from the zone's MEAN cell length and width, and slides that
block over the whole mesh. That holds when the cells are all about the same
size, as PTHA18's ~50 x 50 km cells are. On calabria2 the cells go from 377
to 4042 km2 (the trench is 4 times as long as the deep edge), so the same
2 x 1 block at Mw 7.2 is 798 km2 in one place and 7665 km2 in another
(Strasser: 2390). The moment is still right (slip is set from the real
area), but the area, the slip and, because PTHA18's conditional probability
is proportional to area x convergence, the rate given to each place are not.

**`--rupture-size local`** (pyptha_v12/events.py, `_local_blocks`):

1. at every top-left cell, for each number of rows, the number of columns
   whose summed cell area is closest to Strasser's area; of those, the
   block with the smallest |log area error| + |log aspect-ratio error|
   (rptha's own two criteria, but with the real km of those cells);
2. a placement whose best block would need cells beyond the far edge is
   skipped, as rptha skips placements where its block does not fit (tested
   with one virtual row/column, a copy of the last one);
3. a rupture longer than the zone is the full length, with rows by area
   (rptha's `nlength > nstrike` branch);
4. every cell must be in at least one rupture of every magnitude (rptha's
   blocks always are): a cell left out gets the best block that contains it.

`--rupture-size` flows generate.py -> step 6 (`events.rupture_size` in the
input JSON) -> step 7 (engine), step 7b (HS/VAUS parents) and step 9 (rupture
explorer, which now shows each rupture's own block size). An input without
the key uses rptha's rule. It is not PTHA18's procedure, so with `local` the
event list no longer matches PTHA18's event by event (on PTHA18's own
meshes, 40-99% of the blocks are the same).

**Checks.** Areas of Mw 7.2-8.7 within a factor 1.5 of Strasser:

| mesh | rptha | local | worst factor rptha / local |
|---|---|---|---|
| calabria2 (ours, 20 x 5) | 57.8% | 99.7% | 4.43 / 1.69 |
| 31 PTHA18 meshes (official statistics) | 46-100% | 85-100% | better or equal on 30, flores 1.53 / 1.74 (one corner block) |

No cell is left out of every rupture of a magnitude on any of the 32 meshes.
The smallest rupture is one cell, so where one cell is already larger than
the target (calabria2's centre trench cells, 4042 km2 at Mw 7.2, target
2390) the area cannot be matched. Tests: `pyptha_v12/tests/test_rupture_size.py`.

History note: an earlier `from_scratch_v10` (v9 + the trench ramp rule only)
was deleted on 2026-10-02; the ramp rule was re-implemented in v10_q (see
[Trench ramp rule](#trench-ramp-rule-v10_q)). The present `from_scratch_v10`
does not have it.

### Recipes: v10 and v10_q

Run from `V9/` (the venv is `../../../.venv`). Each folder then runs with its
own `steps/step_total.py`.

```
# Calabria (v10: rupture size only; v10_q: + q). Generated BEFORE the ramp rule
# existed, so their step 1 has no TRENCH_RAMP_RISE_KM (= v9's trench; Calabria
# has no ramp anyway)
python from_scratch_v10/generate.py   calabria  --zone calabria2 --folder calabria2_v10   --ptha false --rupture-size local
python from_scratch_v12/generate.py calabria  --zone calabria2 --folder calabria2_v10_q --ptha false --rupture-size local
# Caribbean: as caribbean2_v9 (step_total.py --skip-hs). Also generated before
# the ramp rule: regenerating caribbean2_v10_q now would trim 68 km of trench
# (use --trench-ramp off to reproduce the folder)
python from_scratch_v10/generate.py   caribbean --zone antilles2 --folder caribbean2_v10   --ptha false --discretizer optimal --rupture-size local
python from_scratch_v12/generate.py caribbean --zone antilles2 --folder caribbean2_v10_q --ptha false --discretizer optimal --rupture-size local
# Hellenic west, with the ramp rule (default) and v10_q's ruptures (--skip-hs, as hellenic_west2_v9)
python from_scratch_v12/generate.py hellenic  --zone hellenic_west2 --folder hellenic_west2_v10_q --ptha false --discretizer optimal --rupture-size local
```

Control runs (engine only, the v9 folders' own inputs with one rule changed;
outputs in `runs/python/<run_name>/`):
`V9/html/v10q_rationale/src/run_controls.py` (kermadectonga2 and kurilsjapan:
`*_control_classic`, `*_control_v10`, `*_control_v10_q`; "classic"
reproduces the v9 folder exactly) and
`runs/python/hellenic_west2_v10q_geometry_classic` (hellenic_west2_v10_q's
input with `rupture_size: rptha`, to separate the ramp rule's effect from
the rupture rule's; its input is in `V9/html/v10q_rationale/inputs/`).

Not regenerated yet with v10_q: `hellenic_east2` (its trench changes slightly,
564 -> 569 km, because it shares hellenic's raster and datum) and
`hellenic2` (segmented, 1557 -> 1487 km of trench).

### Known limitations (v10_q)

- **No rupture smaller than one cell.** Where one cell is larger than
  Strasser's area the area cannot be matched (calabria2's trench cells reach
  4042 km2 against 2390 at Mw 7.2: worst rupture 1.69 times off).
- **The HS generator (step 7b) still assumes equal cells.** rptha's SFFM
  draws each heterogeneous-slip field on the INDEX grid, with one cell
  spacing for the whole footprint: the length and width of the peak cell
  (`step7b_stochastic_slip.py`, `reg_par = kc x peak cell size`; footprint
  cells = Strasser L, W / peak cell size). Inside a v10_q Mw 8.0 rupture the
  cell length varies up to 2.3 times on hellenic_west2_v9 and 2.0 on
  calabria2 (p90; Kurils-Japan 1.17), so there the slip pattern is stretched
  or squeezed in km. Options analysed (not implemented): generate the slip
  with the same spectrum but correlated by real distance on the slab
  (covariance / Karhunen-Loeve on cell centroids), or on a regular km grid
  then averaged onto the cells.
- **Cell shape.** v10_q fixes how ruptures are built from the cells, not the
  cells. Sea-floor deformation (Okada) uses each cell's real, subdivided
  geometry, so a deformed but valid cell is fine there; index-space blocks
  at a sharp corner are still wedge-shaped in km. A mesh-smoothing step 2
  (interior nodes moved on the slab, trench and ends fixed) was prototyped
  but NOT implemented: on hellenic_west2_v9 it took the p90 corner angle
  from 50 to 32 deg and the twist from 0.44 to 0.11 with -0.1% area; on
  calabria2 it changes almost nothing (a fan cannot be meshed regularly with
  rows x columns). After the ramp rule, hellenic_west2's mesh is already at
  p90 31 deg and twist 0.07.
- **Not PTHA18's procedure** with `--rupture-size local`: the event list no
  longer matches PTHA18's event by event, and PTHA18's DART-based
  calibration of its HS/VAUS families (bias adjustment, peak-slip limit) was
  made with rptha's ruptures.
- **Berryman's dip is not used** (as in PTHA18): the dip comes from SLAB2;
  Berryman gives the depth limits (cutoff), coupling, b and Mmax. As a
  check, mesh and Berryman dips agree within a few degrees on Kurils,
  Kermadec, Makran and Antilles, but not on hellenic (7.5-9 deg against
  30-42; rows 5-6 of hellenic_west2 are at 3-4 deg in SLAB2 itself).

### What v9 adds

```
generate.py ... --segmented true
```

splits the zone along strike and runs the full LEVEL 0 tree: the unsegmented
branch and the union of segments each take weight 0.5, and the segments **sum**
within that union (they all occur; they are not alternatives). LEVEL 5 then
draws the epistemic percentiles across segments with a comonotonic copula.

**The method is PTHA18's, checked against rptha line by line** (audit of
2026-09-30, `compute_rates_all_sources.R` and `gcmt_subsetter.R`). A segment is
a stretch of the zone, not a source of its own:

| a segment's ... | as in PTHA18 | where |
|---|---|---|
| scenarios | the zone's whole scenario table; each scenario's conditional probability weighted by the part of it inside the segment (rptha's `is_in_segment`), so a rupture across a boundary takes rate from every segment it touches | engine |
| area, mean dip, Mw_max range | from its own unit sources | engine |
| convergence | area-weighted Bird convergence of its own unit sources | engine |
| edge multiplier | the unsegmented zone's | engine |
| GCMT events | PTHA18's rule on its own cells and unit sources: an event near a boundary can count in both neighbours | step 5 |
| columns | between the segment's two trench end points in Berryman et al. (2015) Table 3.1 (`--segment-boundaries bird`: a run of columns sharing a Bird plate pair) | steps 5, 6 |
| coupling, Mw max observed | its own Berryman et al. (2015) row, the one PTHA18 gives it | step 6 |
| b-value | the zone's (every PTHA18 segment row has the zone's b range) | step 6 |
| the zone's scenario rates | 0.5 x unsegmented + 0.5 x every segment, scenario by scenario, with PTHA18's partial-segmentation percentiles: `scenario_rates_<zone>_source_zone.csv` | engine |

Given PTHA18's own inputs, this reproduces PTHA18's published results exactly:
every segment's logic tree (`validate_v9.py official_segments`, 7 zones) and
every scenario's rate and 5 percentiles on segmented zones, against NCI's
`rate_annual*` (`validate_v9.py events_segmented`, 7 zones: mean within 1e-10, percentiles within 4e-10),
and PTHA18's own GCMT list of every segment (`validate_v9.py gcmt`).

**Where the boundaries come from.** From Berryman et al. (2015) Table 3.1,
the table PTHA18 took its segments from. Its Appendix A defines, for every
segment row, the trench end points (`Left_E_LONG`, `Left_N_LAT`,
`Right_E_LONG`, `Right_N_LAT`; transcribed in
`berryman_params.BERRYMAN_SEGMENT_TRENCH` from the rendered pages 10-13).
`lib/segmentation.py` `berryman_segments()` places each end point on the
mesh's top edge and puts every boundary at the column edge nearest to the end
point two neighbours share; the outer segments are stretched to the mesh's
ends and a segment whose trench lies beyond the mesh is left out. Each
segment then takes its own Berryman row. PTHA18's own table is never read
(neither under `--ptha false` nor `--ptha true`).

Placed on PTHA18's own published meshes, these end points give PTHA18's
boundary indices within 0 to 2 columns on all 18 boundaries the two share
(12 on the same column; `validate_v9.py berryman_segments`, 9 zones):

| zone | Berryman | PTHA18 | boundaries on PTHA18's mesh |
|---|---|---|---|
| kermadectonga2 | 3 | 3 | 22 vs 24, 57 vs 58 |
| kurilsjapan | 2 | 2 | 45 vs 45 |
| izumariana | 2 | 2 | 26 vs 27 |
| ryuku | 2 | 2 | 13 vs 13 |
| newhebrides2 | 4 | 4 | 9, 20, 30 vs 9, 20, 29 |
| solomon2 | 3 | 3 | 19, 28 vs 19, 29 |
| sunda2 | 3 | 4 | 38, 67 vs 39, 67; PTHA18 adds Arakan, not in Table 3.1 |
| southamerica | 6 | 5 | 34, 60, 88, 137 identical; PTHA18 merges Patagonia North and South |
| alaskaaleutians | 6 | 3 | 29, 67 identical; PTHA18 merges the four eastern segments |

v9 keeps Berryman's segments as published rather than PTHA18's merges.

**The alternative, `--segment-boundaries bird`.** Bird (2003)'s plate pairs
(`PB2002_steps.dat`, class `SUB`): each along-strike column is assigned to
the plate pair of the nearest subduction step, and a run of columns sharing a
pair is one segment, which takes the Berryman row(s) its plates name
(`BIRD_SEGMENT_TO_BERRYMAN`). Berryman placed most segment ends where the
plate pair changes, so the two usually agree, but Bird misses a segment whose
plate it does not have: kermadectonga2's Hikurangi is PA\HF, the forearc of
Wallace et al. (2004), so Bird gives 2 segments where Berryman and PTHA18
give 3. Until 2026-09-30 this was the only rule, on the belief that Table 3.1
had names but no coordinates.

**Segmenting a zone that runs past its own boundary.** On kurilsjapan the
SLAB mesh runs on past the Japan Trench onto the Izu-Bonin Trench (5 columns,
no earthquakes, a separate PTHA18 zone). Berryman's Japan segment ends at
34.7 N, so the tail would join the Japan segment (Bird would make it a
segment of its own). Either way it is not the zone: run steps 1-3 once, then
`step_total.py --auto-clip` to cut it (recipe below).

**Each segment's observed maximum (`mw_max_observed`)** is the larger of its own
GCMT maximum (the catalogue threshold, Mw 7.15, when it hosted nothing) and
Berryman's Mmax-min for it: the largest of its own rows (the largest
earthquake thought to have occurred there; kermadectonga2 tonga 8.0, kermadec
8.1, hikurangi 8.0, kurilsjapan 9.0 on both), or, for a Bird segment with no
mapped row, the smallest of the zone's rows. GCMT starts in 1976, so on its own it would tell
kurilsjapan's Kurils-Kamchatka segment that nothing above Mw 8.33 had happened
there (Kamchatka 1952 was Mw 9.0). A segment too small to host that value
(the engine's own Mw_max test) keeps its catalogue value and step 6 warns: on
kurilsjapan that is the Izu-Bonin tail, which is not part of the zone.

**Fixed 2026-09-30 (audit against rptha).** Segmented runs made before this
date are wrong segment by segment:
- every segment took the whole zone's convergence (kermadectonga2: 99 mm/yr
  instead of Tonga 168, Kermadec 91, Hikurangi 30 on PTHA18's own mesh);
- a segment was a slice of the mesh with its own scenarios, so a rupture
  across a boundary had no rate and the zone's scenario rates could not be
  formed;
- each GCMT event was given to exactly one segment, where PTHA18 counts an
  event near a boundary in both;
- every segment took the zone's coupling, where PTHA18 gives each its own
  Berryman row.
Unsegmented runs were not affected by these four.

**Zone values from Berryman, v9 (2026-09-30).** Two zone-level values now
follow Berryman et al. (2015) more closely, and differ from v8's:
- **coupling** is Berryman's own Whole Margin row for the zone where
  Table 3.1 has one (`berryman_params.ZONE_WHOLE_MARGIN`), not the mean of
  its segment rows: kermadectonga2 0.21/0.31/0.72 (v8 0.15/0.25/0.725),
  kurilsjapan 0.67/0.77/0.9, ryuku 0.34/0.44/0.8, sunda2 0.44/0.54/0.79,
  newhebrides2 0.27/0.37/0.73. PTHA18 used the same rows (its coupling now
  equals v9's on all 21 mapped zones).
- **mw_max_observed** is the largest earthquake known on the zone: the
  larger of the zone's largest Berryman Mmax-min and its own GCMT maximum.
  v8 took the dominant segment's Mmax-min, which gave sunda2 7.8 although
  its GCMT holds Sumatra 2004 (Mw 9.03); now 9.03. Also kermadectonga2
  8.0 -> 8.1, ryuku 8.0 -> 8.5, kurilsjapan 9.0 -> 9.12 (Tohoku),
  southamerica 9.0 -> 9.5 (Chile 1960; PTHA18 chose 9.2).
- **b** is unchanged: the mean of Berryman's rows (PTHA18's 0.7-1.2 on
  every zone is its own choice, not Berryman's).
So an unsegmented v9 run no longer equals v8's on these zones; what
`validate_v9.py v8parity` now checks is that v9's ENGINE, given v8's own
input, still gives v8's outputs bit for bit.

**Report labels, v9 (2026-09-30).** On a segmented run the rate chart's
percentile band and the full rate table are the whole zone's (the LEVEL 0
mix), while the other blocks describe the unsegmented branch: the report now
says so above the chart and the table, and labels the "this run vs PTHA18"
rate row "(unsegmented branch)". The attribution box under that table no
longer names a cause when the Mw 7.2 rates agree within 1%. A box under the
report's title now says first whether the run is SEGMENTED (with its segments
and where their boundaries came from) or UNSEGMENTED.

**Unsegmented view, v9 (2026-10-01).** A segmented run already computes the
unsegmented branch in full, so step 9 now also writes
`report_unsegmented.html`, the report an unsegmented run of the zone gives,
and both pages open with a switch (Segmented / Unsegmented only) that links
them. The one thing missing on disk was the unsegmented branch's own
percentile band (the segmented run's band is the zone mix), so on a segmented
run the engine also writes `exceedance_rate_percentiles_unsegmented.csv`,
with the exact call an unsegmented run makes. `validate_v9.py unseg_view`
checks that the unsegmented branch of `<zone>_v9seg` (tables, curve, band,
for this run and the official one) is bit-identical to `<zone>_v9`: PASS on
kermadectonga2 and kurilsjapan. The page differs from `<zone>_v9/report.html`
only in the switch, the box under the title and the folder name. An
unsegmented run is unchanged: no switch, no extra file, `report.html` byte for
byte what it was.

**Public-data audit, v9 (2026-09-30).** Every step was compared again with
rptha's R code, now on the code paths only a from-scratch run uses (the
validation checks run PTHA18's own inputs). Fixed:
- step 7b (HS/VAUS): redraws a field until every edge of its window has slip,
  clips the random L/W/kcx/kcy at 2 sd, uses one random stream per
  (magnitude, placement, realisation) and rptha's max(15, ceiling(200/n))
  fields per placement, as `sffm_make_events_on_discretized_source` and
  `make_all_earthquake_events.R` do. No rate changes (HS/VAUS never feed the
  logic tree);
- the coupling of a zone Berryman does not treat is Berryman's own default
  0.3/0.5/0.7, as PTHA18 applies it (ReportPTHA 3.7.2.3), not [0.1, 1.3];
- step 6 no longer crashes on such a zone (Mw max observed falls back to its
  largest GCMT event);
- the GCMT window is [1976-01-01, 2017-03-01), PTHA18's;
- the engine clears its own old files in `runs/python/<run_name>/` before a
  run (a re-run with a different segmentation copied the old segments'
  files into `outputs/`).
Kept, as decided: b from Berryman's rows (PTHA18 uses 0.7-1.2 on every zone;
identical on kermadectonga2 and puysegur2), Mw max observed as the largest of
Berryman and GCMT (southamerica 9.5 where PTHA18 chose 9.2), Bird's public
catalogue for convergence (Bird+Griffin with `--convergence bird-griffin`).
Latent, not reachable by a thrust zone built by step 6: rake is 90 on the
shapefile path and in step 5's strike filter. A file-open hook over a
segmented kermadectonga2 run, steps 1-7, 7b and 9, logged no read of any
PTHA18 or rptha file (html/docs/data_provenance.html).

**Segment boundaries from Berryman, v9 (2026-09-30).** The default
boundaries are now Berryman et al. (2015) Table 3.1's segment end points
instead of Bird's plate pairs (`--segment-boundaries bird` keeps the old
rule). kermadectonga2_v9seg gets PTHA18's three segments (tonga 1-23,
kermadec 24-57, hikurangi 58-73 against PTHA18's 1-24, 25-58, 59-72) and its
segmented answer moves from -8% to -3% of PTHA18's at Mw 7.2 and from +37% to
+17% at Mw 9.2; segmenting now changes it by +12% / -24%, PTHA18's own
+15% / -25%. kurilsjapan_v9seg's Japan segment gets PTHA18's 8 earthquakes
(10 with Bird) and its rate is within 2.3% of PTHA18's (+15% with Bird).

### Clip windows: the window's edge is never trench (2026-10-02)

With `--clip` / `CLIP_BBOX` / `--auto-clip`, step 1 used to accept the window's
own straight edge as trench wherever it cut through the slab shallow enough
and facing up dip. Half of hellenic_east2 was then measured from that false
trench (14-22 km deep instead of 11-15), and on its nearly flat slab the
contours looped and 4 cells folded. Now a boundary vertex next to slab that the
window removed is never trench (`lib/slab_contours.py`, `find_trench(cut=)`);
step 1's log reports the km of window edge. Effect, while hellenic_east2 still
used its window (it now uses --trench-ends, below): 11 x 7, 58%
near 50 x 50, 4 folded -> 8 x 7, 98%, 0; antilles2 75% -> 84% (rate +0.6% at
Mw 7.2). Unclipped zones and kurilsjapan: identical. Test:
`test_step1_clip_line_through_the_slab_is_never_trench`.

### Zones cut at Berryman's segment ends: `--trench-ends` (2026-10-02)

The Hellenic zones no longer use a clip window at all. A lon/lat rectangle
cuts the slab along straight meridians, which is not where the zones end:
hellenic_west2's western edge (19.512E) ran parallel to its own trench and
left the real trench outside. Now `generate.py --trench-ends LON1,LAT1,LON2,LAT2`
(default for hellenic_west2, hellenic_east2 and hellenic2:
`berryman_params.ZONE_DEFAULT_TRENCH_ENDS`, from Berryman et al. 2015 Table
3.1 rows 48-49) makes step 1 build the trench, datum and contours on the WHOLE
SLAB raster and cut every contour along the down-dip line through each point
(`slab_contours.below_trench_contours(trench_ends=)`, `_cuts_at_points`).
Berryman's points lie on the bathymetric Hellenic Trench, 20-220 km down dip
of SLAB2's up-dip edge, so the cut is the line of steepest descent that passes
closest to each point (17-31 km), not the nearest trench point. The western
segment ends at 25.288E and the eastern one starts at 25.228E (34.202N): the
two zones share the midpoint, 25.258E, so they tile exactly.

| zone (re-run) | old rectangle | Berryman ends |
|---|---|---|
| hellenic_west2_v9 | 17 x 7, 58% near 50 x 50, 193,556 km2 | 20 x 8, 65%, 253,725 km2 |
| hellenic_east2_v9 | 11 x 7, 58%, 4 folded (8 x 7, 98% with the clip rule) | 12 x 6, 75%, 129,200 km2 |
| hellenic2_v9seg (whole arc, `--segmented true`) | did not exist | 32 x 7, 70%, 382,656 km2 |

`hellenic2` was added to `ZONE_BERRYMAN_SEGMENTS` (west / east), with
plate pair AF\AS for both rows and the Hellenic Whole Margin row for the zone
values. Test: `test_step1_trench_ends_cut_the_zone_at_the_given_points`.

Step 3, for these three zones only (`berryman_params.zone_plate_pairs`), matches
each column to the nearest Bird (2003) step **between Berryman's plates**
(AF-AS = Bird AS/AF), not to the nearest step of any boundary. SLAB2's trench
lies 120-300 km from Bird's here, so the plain nearest step was sometimes another
boundary: hellenic_west2's four north-west columns took Calabria's EU/AF (7 mm/yr)
and the EU-AF boundary north of Kefalonia (4 mm/yr), which also raised step 3's
plate-boundary warning, although Bird's own Hellenic AS/AF runs to 37.45N (28
mm/yr); hellenic_east2's eastern columns took the Anatolia-Africa steps near
Cyprus. Zone convergence: west 30.5 -> 37.1, east 29.5 -> 33.8, hellenic2 30.4
-> 35.9 mm/yr; no plate-boundary warning left. Every other zone is unchanged.
Test: `test_hellenic_zones_match_only_their_own_bird_plates`.

### Recipes: segmented runs and their official comparison

The whole topic in one page, with both worked examples: [html/docs/segmentation.html](html/docs/segmentation.html).

```
# a segmented run (steps 1-9; step 8 is the official PTHA18 comparison)
.venv/Scripts/python.exe from_scratch_v12/generate.py kermadec --zone kermadectonga2 --folder kermadectonga2_v9seg --ptha false --segmented true
.venv/Scripts/python.exe kermadectonga2_v9seg/steps/step_total.py --skip-hs

# kurilsjapan: steps 1-3 first (step 3 flags the Izu-Bonin tail), then the clipped run
.venv/Scripts/python.exe from_scratch_v12/generate.py kuril --zone kurilsjapan --folder kurilsjapan_v9seg --ptha false --segmented true
.venv/Scripts/python.exe kurilsjapan_v9seg/steps/step1_fetch_slab2.py
.venv/Scripts/python.exe kurilsjapan_v9seg/steps/step2_build_grid.py
.venv/Scripts/python.exe kurilsjapan_v9seg/steps/step3_convergence.py
.venv/Scripts/python.exe kurilsjapan_v9seg/steps/step_total.py --skip-hs --auto-clip

# only the official comparison, on a folder whose steps 1-7 have run
.venv/Scripts/python.exe kermadectonga2_v9seg/steps/step8_official.py
.venv/Scripts/python.exe kermadectonga2_v9seg/steps/step9_report.py
```

Step 8 needs R with rptha (it extracts PTHA18's trees from the saved session,
1.34 GB, downloaded once). On a segmented example it runs **PTHA18's own
segmented model**, into `runs/python/<zone>_official_segmented/`, copied to
`<folder>/outputs_official/`, with PTHA18's segments listed in
`outputs_official/ptha18_reference/official_segments.json`; on an unsegmented
example, PTHA18's unsegmented branch into `runs/python/<zone>_official/`. The
report's LEVEL 0 section then shows both segmentations (a bar and
`figures/segments_map.png`), a table per model, this run against PTHA18 for
the unsegmented branch, the union and the mix, and a chart. The unsegmented
twin (the same command without `--segmented true`, folder `kermadectonga2_v9`)
gives the unsegmented comparison; the four folders `kermadectonga2_v9`,
`kermadectonga2_v9seg`, `kurilsjapan_v9`, `kurilsjapan_v9seg` were made exactly
like this, and html/segmented_vs_unsegmented.html (in the package root's
`html/`) compares them.

---

## Contents

0. v10 / v10_q: [versions table](#from_scratch_v12), [Trench ramp rule](#trench-ramp-rule-v10_q),
   [What v10_q changes](#what-v10_q-changes), [What v10 changes](#what-v10-changes),
   [Recipes: v10 and v10_q](#recipes-v10-and-v10_q), [Known limitations (v10_q)](#known-limitations-v10_q)
1. [TL;DR](#1-tldr)
2. [Quick start](#2-quick-start)
3. [Folder layout](#3-folder-layout)
4. [How it works (the big picture)](#4-how-it-works-the-big-picture)
5. [generate.py: every option](#5-generatepy-every-option)
6. [`--ptha`: exactly what it changes](#6---ptha-exactly-what-it-changes)
7. [`--discretizer`: exactly what it changes](#7---discretizer-exactly-what-it-changes)
   - [7b. History: the v5, v6 and southamerica fixes](#7b-history-the-v5-v6-and-southamerica-fixes)
   - [7e. What v8 changed](#7e-what-v8-changed)
8. [The ten steps, one by one](#8-the-ten-steps-one-by-one)
9. [Recipes: commands for every zone](#9-recipes-commands-for-every-zone)
10. [Customising a run](#10-customising-a-run)
11. [Verification](#11-verification)
12. [Known caveats](#12-known-caveats)
13. [Glossary](#13-glossary)
14. [Further reading](#14-further-reading)

---

## 1. TL;DR

- **What it is:** one command (`generate.py`) writes a new example folder
  holding nine numbered Python scripts (`step1` ... `step9`, plus `step7b`)
  and a `step_total.py` that runs them all.
- **What the scripts do:** download the fault geometry (SLAB), draw PTHA18's
  "depth below the nearby trench" contours with square-cut ends, cut them
  into unit sources,
  compute the plate convergence (Bird 2003), download and filter the
  earthquake catalogue (GCMT), run the full PTHA18 logic tree (LEVELs 0 to
  5), and write an HTML report.
- **How faithful it is (v8):** fed PTHA18's OWN inputs, every stage gives
  PTHA18's own published output: the mesh (14 of 14 zones), the unit-source
  table, the Bird convergence (7 of 7), the GCMT event list (8 of 8), the
  32,000 logic-tree branches (4 of 4) and the rate of every earthquake
  scenario with its percentiles (4 of 4, to 1e-10 or better). See
  [section 11](#11-verification).
- **Where the physical inputs come from:** Berryman et al. (2015) for
  coupling, b-value, Mw_max_observed and the seismogenic depth cutoff; SLAB
  for the geometry and the trench; Bird (2003) for convergence; GCMT for the
  observed earthquakes. Nothing is copied from PTHA18's own results unless
  you allow it (`--ptha true`) or run the optional comparison (step 8).
- **The two switches that matter most:**
  - `--ptha false` means "public data only": steps 1-7, 7b and 9 read no
    file of PTHA18 or of rptha (50 km cells, the row count from the
    geometry, Bird's public catalogue, Strasser/30 GPa); only step 8, the
    official comparison run, does. **Use this for a genuinely independent
    run, or for a zone PTHA18 never meshed.**
  - `--discretizer optimal` (the default) means "rptha's own mesh, exactly
    as PTHA18 builds it, repaired only if it is really broken" (folded,
    collapsed or overlapping cells).
- **The recommended command** (the one every example used, `*_v8` and `*_v9`):

  ```
  .venv/Scripts/python.exe from_scratch_v12/generate.py <region> --zone <ptha18_zone> --folder <new_folder> --ptha false --discretizer optimal
  .venv/Scripts/python.exe <new_folder>/steps/step_total.py
  ```

- **Result:** `<new_folder>/report.html`, plus every intermediate file on
  disk.

---

## 2. Quick start

All commands are run **from `ptha18_logic_tree_test/`** (the parent of this
folder), with the project's virtual environment.

```
# 0. (optional) see which regions / PTHA18 zones exist
.venv/Scripts/python.exe from_scratch_v12/list_regions.py

# 1. generate the example folder (here: Puysegur, fully independent of PTHA18's mesh)
.venv/Scripts/python.exe from_scratch_v12/generate.py puysegur --zone puysegur2 --folder puysegur2_v9 --ptha false --discretizer optimal

# 2. run the whole pipeline (step 8 needs R; skip it if you do not have R + rptha)
.venv/Scripts/python.exe puysegur2_v9/steps/step_total.py --skip-official

# 3. open the report
start puysegur2_v9/report.html
```

Timing on this machine, with the SLAB grid and GCMT files already cached:
puysegur2 about 1 minute for the whole pipeline (including step 7b), the
largest zone (southamerica) about 4 minutes without step 7b. The first run
of any zone takes longer, because steps 1 and 4 download the SLAB grid and
the GCMT catalogue.

---

## 3. Folder layout

### 3.1 This folder (the generator)

```
from_scratch_v12/
|
|-- README.md                  <- this file
|-- generate.py                <- ENTRY POINT: writes a new example folder
|-- list_regions.py            <- ENTRY POINT: lists the 27 SLAB regions + PTHA18 zone guesses
|
|-- lib/                       <- helper modules (imported by generate.py and the steps)
|   |-- slab_contours.py            step 1: the trench, "depth below the nearby trench", contours, checks (v8); trench ramp rule (_ramp_trim, v10_q)
|   |-- bird_convergence.py         steps 3, 8, 9: Bird (2003) matching and PTHA18's Bird model (v8)
|   |-- slab2_catalog.py            the 27 SLAB2.0 regions (ScienceBase ids, prefixes), name matching
|   |-- berryman_params.py          Berryman et al. (2015) tables: cutoff, coupling, b, Mw_max_obs
|   |-- official_geometry_params.py SLAB grid lookup/download, widths, row counts, official area/dip
|   |-- hs_official_compare.py      step 7b's statistical comparison with PTHA18's HS/VAUS catalogues
|   `-- plot_meshes.py              mesh figures (also a CLI, see section 10.5)
|
|-- templates/                 <- the step scripts, with {{PLACEHOLDERS}} filled by generate.py
|   |-- step1_fetch_slab2.py.tmpl ... step9_report.py.tmpl, step7b_stochastic_slip.py.tmpl, step7c_variable_mu.py.tmpl (v11)
|   |-- step_total.py.tmpl
|   `-- RUN.html.tmpl
|
|-- pyptha_v12/                 <- the Python rptha port (v8's, plus LEVEL 0 (v9), cell-by-cell ruptures (v10) and q (v10_q))
|   |-- contour_discretisation.py   the mesh builder (optimal / lm / mid), rptha's optimiser, mesh check
|   |-- grid_cache.py               step 2's mesh saved with a fingerprint, reused by steps 7 and 9 (v8)
|   |-- events.py                   ruptures: rptha's rule, _local_blocks (v10, --rupture-size local), coverage_weights (v10_q, q)
|   |-- unit_sources.py, rates.py, moment_balance.py, logic_tree.py, ...
|   |-- examples/                   standalone usage examples
|   |-- hs_vaus_rates.py            v10_q: PTHA18's HS/VAUS rates from the FAUS rates (peak-slip limit + weights; v11: variable_mu curves)
|   |-- variable_mu.py              v11: rigidity curve, variable_mu_Mw, conditional ECDF (PTHA18's variable shear modulus)
|   |-- mesh_file.py                v12: reads an external quadrilateral mesh (--mesh-file) into the node array
|   `-- tests/                      187 tests; test_mesh_file.py (v12); test_variable_mu.py (v11); test_v8_fidelity.py checks against PTHA18's own files; test_rupture_size.py (v10, v10_q), test_trench_ramp.py (v10_q)
|
|-- python_logic_tree_v12/      <- the logic-tree engine (LEVELs 0-5), used by steps 7 AND 8; reads events.rupture_size and applies q with "local"
|   `-- run_logic_tree.py
|
|-- validation/
|   |-- validate_v9.py              re-runs every check of html/docs/validation.html against PTHA18's files
|   |-- recover_peak_slip_weights.py  v10_q: recovers PTHA18's HS/VAUS weight table from its published rates and checks it (v11: 4 curves)
|   `-- validate_v11.py              v11: variable_mu_Mw, the engine's variable_mu rates and HS/VAUS rates vs PTHA18's files
|
|-- data/slab1/, data/slab2/   <- empty; a .grd put here by hand is reused by every run folder
|-- data/ptha18_peak_slip_quantile_weights.csv  <- v10_q: PTHA18's HS/VAUS weights, recovered (see validation/)
|
`-- html/docs/                 <- the documentation (open index.html in a browser)
    `-- figures_src/                the scripts that make its figures
```

Folders generated from here import **only** from `from_scratch_v12/`.

### 3.2 A generated example folder

`generate.py` creates the folder next to this one, in
`ptha18_logic_tree_test/<folder>/`:

```
<folder>/
|-- README.md, RUN.html          what this example is + how to run it
|-- steps/                       step1 ... step9, step7b, step7c (v11), step_total (zone-specific, ready to run)
|-- data/
|   |-- slab2/                   SLAB grid (downloaded), unit_source_grid.npy + its fingerprint (step 2)
|   |-- step1_contours_info.json step 1: length, pieces and edge contact of every contour level
|   |-- gcmt/                    raw GCMT .ndk files, full catalogue CSV, <zone>_gcmt_subset.csv
|   |-- convergence.txt          step 3: the zone's convergence (mm/yr, horizontal)
|   |-- convergence_per_column.json   step 3: Bird values per along-strike column (v8)
|   `-- convergence_per_cell.npy
|-- inputs/
|   |-- geometry/<zone>_slab2_contours.shp   step 1 contours
|   |-- input_<zone>_scratch.json            step 6: THE input to the logic tree (editable)
|   `-- input_<zone>_official.json           step 8: this example's copy of the official input
|-- figures/                     step1_trench_and_contours.png, mesh maps and profiles (steps 1, 2, 3, 9)
|-- outputs/                     step 7 results (from-scratch run)
|-- outputs_official/            step 8 results (official PTHA18 run), if step 8 ran
`-- report.html                  step 9
```

A segmented folder (`--segmented true`) has the same layout and files, plus:
`data/gcmt/<zone>_gcmt_subset_<segment>.csv` and `segments.json` (step 5),
a `segments` block in the input JSON (step 6), one set of
`scenario_rates_`, `conditional_prob_`, `integrated_slip_`,
`logic_tree_branches_` files and five figures per segment, and
`scenario_rates_<zone>_source_zone.csv`, the zone's own per-scenario rates
(step 7), the same for PTHA18's segments in `outputs_official/` with
`ptha18_reference/official_segments.json` (step 8), and
`exceedance_rate_percentiles_unsegmented.csv`, the unsegmented branch's own
band, in `outputs/` and `outputs_official/` (steps 7 and 8), and
`figures/segments_map.png`, `outputs/exports/scenario_rates_source_zone*.csv`
and `report_unsegmented.html` (step 9): 64 more files on kermadectonga2. Every one, with its columns:
[html/docs/reference.html#segmented-files](html/docs/reference.html#segmented-files).

---

## 4. How it works (the big picture)

```
                     generate.py <region> --zone Z --folder F --ptha P --discretizer D [--clip W]
                                                |
                     fills templates/*.tmpl with Z, F, P, D, W, SLAB prefix, ScienceBase id
                                                |
                                                v
   +-------------------------------------- F/steps/ ---------------------------------------+
   |                                                                                        |
   |  GEOMETRY (LEVEL 0 input)          CONVERGENCE (LEVEL 1)      SEISMICITY (LEVEL 3)      |
   |  step1  SLAB grid -> trench +      step3  Bird (2003) per     step4  download GCMT      |
   |         contours below the trench         column -> mm/yr     step5  filter to zone     |
   |  step2  contours -> unit sources                                                        |
   |         (area, mean dip)                                                                |
   |          \                               |                        /                      |
   |           \______________________________|_______________________/                       |
   |                                          v                                               |
   |                     step6  assemble input_<zone>_scratch.json                            |
   |                            (+ Berryman: coupling, b, Mw_max_observed)                    |
   |                                          v                                               |
   |                     step7  the engine: LEVELs 0-5 -> outputs/                             |
   |                                          v                                               |
   |   step8 (optional, R)  official PTHA18 input, same engine -> outputs_official/           |
   |                                          v                                               |
   |                     step9  report.html                                                    |
   +----------------------------------------------------------------------------------------+
```

step7 also feeds an optional, additive branch that never changes any of step 7's own numbers:

```
step7 outputs/  --(re-reads the same FAUS event table)-->  step7b (optional)
                                                                |
                                          HS + VAUS variants of the FAUS ruptures
                                                                |
                                                                v
                               outputs/hs_slip_fields/, outputs/vaus_slip_fields/
```

**Where each number comes from:**

| quantity | source | step |
|---|---|---|
| fault surface | SLAB2.0 grid (zones ending in `2`) or SLAB1.0 grid (the rest) | 1 |
| trench (the 0 km contour) and the depth datum | the SLAB grid itself: its shallow edge, measured locally | 1 |
| seismogenic depth cutoff | Berryman et al. (2015) Table 3.1, dominant segment | 1 |
| unit-source grid, total area, mean dip | rptha's discretiser, reproduced exactly | 2 |
| tectonic convergence (mm/yr), per column and for the zone | Bird (2003), matched the way PTHA18 does it | 3 |
| observed earthquakes | GCMT, 1976-01-01 to 2017-03-01, PTHA18's selection rule | 4, 5 |
| coupling, b-value, Mw_max_observed (min/pref/max) | Berryman et al. (2015) | 6 |
| logic-tree resolution (20 coupling x 20 b x 40 Mw_max x 2 GR types = 32 000 branches) | fixed in step 6 | 6 |

**The logic tree (step 7) in one line per level:**

- **LEVEL 0, segmentation:** without `--segmented true`, no segments are declared and the zone is modelled unsegmented (weight 1). With it, the zone is split along strike at Berryman et al. (2015) Table 3.1's segment end points (or Bird (2003)'s plate pairs, `--segment-boundaries bird`): the unsegmented branch takes weight 0.5, the union of segments 0.5, and each segment carries the full union weight because the union is a **sum** over segments, not a choice between them (report Section 3.7.6). LEVELs 1-4 below then run once per representation.
- **LEVEL 1, branches:** every combination of coupling x b x Mw_max x Gutenberg-Richter type (truncated 0.7, characteristic 0.3).
- **LEVEL 2, moment balance:** for each branch, solves the GR `a` value so that the long-term seismic moment equals area x slip rate x rigidity, with slip rate = convergence / cos(mean dip) x coupling. **This is where the rate curves are born.**
- **LEVEL 3, Bayesian update:** re-weights the branches against the observed GCMT events (prior to posterior). It changes weights, not curves.
- **Scenario probabilities:** within each magnitude, a scenario's share of the rate is proportional to its area times the Bird convergence under it (PTHA18's model for Bird zones).
- **LEVEL 4, edge correction:** fits how much more likely scenarios touching the zone's ends are, so that long-term slip follows the Bird convergence shape.
- **LEVEL 5, percentiles:** Monte Carlo (40 000 samples, comonotonic copula) over branch weights, giving the 2.5/16/50/84/97.5 % curves.

---

## 5. generate.py: every option

```
.venv/Scripts/python.exe from_scratch_v12/generate.py NAME [--zone ZONE] [--folder FOLDER]
                                                            [--id SCIENCEBASE_ID]
                                                            [--ptha {true,false}]
                                                            [--discretizer {optimal,lm,mid}]
                                                            [--columns {trench,average}]
                                                            [--cell-size {mean,strasser}] [--cell-k K]
                                                            [--rupture-size {rptha,local}]
                                                            [--trench-ramp {on,off}]
                                                            [--segmented {true,false}]
                                                            [--clip LON_MIN,LON_MAX,LAT_MIN,LAT_MAX]
```

| argument | default | what it does |
|---|---|---|
| `NAME` (required) | none | SLAB region to use, matched loosely against the 27 region names and 3-letter prefixes: `kermadec`, `puysegur`, `kuril`, `"south america"`, `sam`, ... |
| `--zone` | guessed from NAME | exact PTHA18 `sourcename` from `sourcezone_parameters.csv` (`kermadectonga2`, `puysegur2`, `izumariana`, ...), or a zone with a Berryman mapping (`calabria2`, `antilles2`). **Always pass it when a region maps to several zones**, otherwise generate.py stops and lists them. |
| `--folder` | `<name>_v10_q` | name of the folder to create under `ptha18_logic_tree_test/`. **Never overwrites:** if it exists, `_2`, `_3`, ... is appended. |
| `--id` | none | ScienceBase item id, to pick the region explicitly instead of by name (see `list_regions.py`). |
| `--ptha` | `true` | whether steps 1-7 and 9 may read PTHA18's files (width, row count, scaling). See [section 6](#6---ptha-exactly-what-it-changes). |
| `--discretizer` | `optimal` | how the contours are cut into unit sources. See [section 7](#7---discretizer-exactly-what-it-changes). |
| `--columns` | `trench` | v8.2: how many unit sources along strike the `optimal` mesh gets. `trench`: rptha's and PTHA18's rule, trench length / 50 km. `average`: the average row length / 50 km, **not PTHA18's procedure**, for a zone whose mesh tapers far more than any PTHA18 mesh (a tight arc: calabria2 20 -> 12 columns, median cell length 26 -> 45 km). Step 2 prints the taper next to PTHA18's maximum and suggests `average` when it is exceeded. See [v8.2](#v82). |
| `--cell-size` | `mean` | v12: how big the unit sources may be. `mean`: rptha's and PTHA18's rule, ~50 x 50 km on average (some cells much larger on a fan or a zone of varying width). `strasser`: **no** cell longer or wider than the Mmin rupture (Strasser, Mw 7.2: 54 x 44 km) divided by `--cell-k`; same mesher, more rows and columns, **not PTHA18's procedure**, 3-7 times more unit sources. Needs `--discretizer optimal`; use with `--rupture-size local`. See [`--cell-size strasser`](#--cell-size-strasser-2026-10-07). |
| `--cell-k` | `1.5` | v12, with `--cell-size strasser`: the cap is the Mmin rupture / K. 1.5: cells of at most 36 x 29 km; 2: 27 x 22 km. |
| `--rupture-size` | `rptha` | v10: how many cells each uniform-slip rupture gets. `rptha`: rptha's and PTHA18's rule, one block per magnitude from the zone's mean cell size. `local`: a block per placement from the real km of the cells there, **not PTHA18's procedure**, for a mesh whose cells differ a lot in size (calabria2: 377 to 4042 km2); v10_q: with `local`, the engine also multiplies each rupture's conditional-probability weight by q (rupture-overlap correction). Written to the input JSON as `events.rupture_size`. See [What v10 changes](#what-v10-changes) and [What v10_q changes](#what-v10_q-changes). |
| `--mesh-file` | none | v12: an external quadrilateral mesh (one line per unit source, its 4 corners as lon lat depth) used instead of SLAB for steps 1-2; must be structured (rows x columns). See [What v12 changes](#what-v12-changes). |
| `--mesh-depth-units` | `auto` | v12, with `--mesh-file`: `m` or `km`; `auto` = metres if any \|depth\| > 200. The sign is detected. |
| `--variable-mu` | `off` | v11: `on` makes `step_total.py` run step 7c, PTHA18's variable shear modulus rates (`variable_mu_*`) next to the constant ones; `off`: constant 30 GPa only, step 7c does nothing unless run with `--force`. See [What v11 changes](#what-v11-changes). |
| `--trench-ramp` | `on` | v10_q: a trench end that climbs in depth like a ramp (more than 3 km deeper than the trench's normal depth) is the edge of the SLAB data, not trench, and is trimmed (at most 100 km) before the local datum is built. Step 1's `TRENCH_RAMP_RISE_KM` (3.0, or None with `off` = v9's trench). Changes hellenic (71 km), caribbean/antilles (68 km), kermadectonga2 (11 km); other zones identical. See [Trench ramp rule](#trench-ramp-rule-v10_q). |
| `--segmented` | `false` | v9: `true` runs LEVEL 0 with segments (unsegmented branch 0.5, union of segments 0.5, segments summed within the union), their boundaries set by `--segment-boundaries`. `false`: every zone unsegmented, v8's model (v9's engine gives v8's outputs bit for bit from v8's own input; from a new step 6 the numbers differ where v9 changed the zone's Berryman values). See [What v9 adds](#what-v9-adds). |
| `--segment-boundaries` | `berryman` | v9, with `--segmented true`: `berryman` puts the boundaries at the trench end points Berryman et al. (2015) Table 3.1 gives every segment (the segments PTHA18 used); `bird` at Bird (2003)'s plate-pair changes. Public data either way. |
| `--convergence` | `bird` | which plate-boundary table step 3 reads, both in `data/bird/`: `bird`, Bird (2003)'s public catalogue; `bird-griffin`, the table PTHA18 used (Bird plus Jonathan Griffin's traces). They differ only where PTHA18 used Griffin's traces (eastern Indonesia, New Guinea); see `html/docs/convergence_sources.html`. |
| `--clip` | none | v8: use only this lon/lat window of the SLAB grid, for a SLAB region that covers more than the zone. Lesser Antilles: `--clip 297,307,5,18.5`. Longitudes in the grid's own convention. |

**What generate.py prints before writing anything** (check it, it is the
cheapest moment to catch a wrong zone): the SLAB region and prefix, the
PTHA18 zone it matched, your flags, the Berryman inputs, and (with
`--ptha true` only) the official area, mean dip and mesh size when the
official table is available locally.

**Which SLAB product is used** is not a flag. It follows PTHA18's own rule:
a zone name ending in `2` (`kermadectonga2`, `puysegur2`, `makran2`, ...) uses
SLAB2.0; any other zone (`kurilsjapan`, `cascadia`, `southamerica`, ...) uses
SLAB1.0.

---

## 6. `--ptha`: exactly what it changes

**Question it answers:** "May steps 1-7 and 9 read PTHA18's or rptha's
files?" With `false` none of them does; the example runs were checked with a
hook that logs every such file any Python process opens, and only step 8's
processes appear.

| | `--ptha true` (default) | `--ptha false` (recommended for independent runs) |
|---|---|---|
| generate.py zone name | from `sourcezone_parameters.csv` | from `lib/berryman_params.py` |
| **step 2** unit-source width | per zone, from `sourcezone_parameters.csv` (e.g. puysegur2 = 35 km) | **50 km** for every zone (PTHA18's general value) |
| **step 2** number of down-dip rows | fixed to the official row count when the official table exists | inferred by rptha's width-based rule (or `N_DOWNDIP_OVERRIDE`, see 10.2), at least 2 |
| **step 3** Bird table | not set by `--ptha`: see `--convergence` | not set by `--ptha`: see `--convergence` |
| **step 6** scaling relation, shear modulus | PTHA18's row for the zone | Strasser et al. (2010), 30 GPa |
| steps 2, 3, 5 comparison lines in the log | printed | not printed (the report shows the comparison, from step 8) |
| works for a zone PTHA18 never meshed? | falls back to the `false` behaviour when the table is missing | **yes, always** |

**What `--ptha` does NOT change:** step 1's contours (v8: always PTHA18's
recipe, "depth below the nearby trench" every 5 km), the Berryman inputs,
the SLAB product, GCMT (steps 4-5), and step 8. Step 9 reads PTHA18 only
from `outputs_official/`, that is from step 8.

---

## 7. `--discretizer`: exactly what it changes

It chooses the algorithm step 2 (and, through the JSON, steps 7 and 9) uses
to turn the depth contours into the grid of unit sources.

| value | what it is | when to use it |
|---|---|---|
| `optimal` **(default)** | rptha's own mesh (`lm` below), checked for defects: folded (bow-tied) cells, cells under 2 % of the median area, columns overlapping by more than 0.1 % of the area, invalid nodes. **No defect: the mesh is kept unchanged.** Otherwise three repairs are tried in order (rptha with a nearest-point start; contours trimmed to a common span; both) and the first defect-free one is kept; if none is, the one with fewest defects is kept and a WARNING is printed. | Always, unless you are comparing methods. |
| `lm` | rptha's method exactly as PTHA18's template (`make_initial_downdip_lines.R`) calls it, with no check or repair. v8 reproduces rptha's `minpack.lm::nls.lm` optimiser step for step. | To reproduce PTHA18's automatic mesh method with no safety net. |
| `mid` | rptha's older, deprecated method (evenly spaced cuts). PTHA18 does not use it. | Comparison or diagnosis only. |

Measured with v8 (`--ptha false`), the hardest zones built from scratch:

| zone | mesh | broken cells | mesh kept | ends cut by step 1 |
|---|---|---|---|---|
| calabria2 | 20 x 5 | 0 | rptha's own | west (crooked), north-east (taper) |
| caribbean2 (whole region) | 47 x 4 | 0 | rptha's own | south (crooked) |
| antilles2 (`--clip 297,307,5,18.5`) | 24 x 6 | 0 | rptha's own | south (crooked), north (taper) |
| southamerica | 118 x 4 | 0 | rptha's own | south (crooked) |

Every end of these meshes is now within 39 deg of square, inside the range
of PTHA18's own hand-made ends (2-42 deg). Southamerica needed a repair
(1 folded and 3 collapsed cells) only while its crooked south end was
there; with step 1's end cut, rptha's own mesh is clean.

And fed PTHA18's own contours (NCI `DATA/SOURCEZONE_CONTOURS.zip`), `optimal`
gives PTHA18's own published mesh on all 14 zones tested
(`validation/validate_v9.py mesh`).

The discretizer choice is written into the input JSON (`geometry.discretizer`),
and step 2's mesh is saved with a fingerprint, so steps 7 and 9 use **the same
mesh** step 2 checked.

---

## 7b. History: the v5, v6 and southamerica fixes

Kept for reference; each is still in the code unless marked superseded.

- **v5, three engine fixes**: spherical
  geodesy where rptha is spherical, `computational_increment = 0.02` as in
  PTHA18's driver, and the edge-multiplier degeneracy check evaluated at 0.
- **v4/v5, tip folds**: the reason
  `optimal` first existed. **Superseded in v8:** the folds came from contours
  whose shallow ends did not face each other, which v8's step 1 no longer
  produces; `optimal` now keeps rptha's mesh unless it is broken.
- **v6, large-zone safety net**:
  a retry that scrambled the southamerica mesh (1767 bow-ties) and a 16-degree
  hole in SLAB1.0's shallowest contour. **Superseded in v8:** the optimiser
  now has no timeout and uses rptha's own retry, and step 1 no longer uses
  the shallowest-contour guard.
- **v6, nine fidelity fixes**: GCMT hypocentre-or-centroid test (now replaced
  by the full R rule, 7e), flat LEVEL 4 target when no convergence profile is
  given (v8 now gives the Bird profile, 7e), HS peak-location window and
  spherical centroid, the Kermadec/Tonga b-value, the normal-fault Mw_max cap,
  per-zone shear modulus and scaling relation, the optimiser's retry proxy
  (superseded by the exact rptha retry), and spherical transect resampling.

---

## 7e. What v8 changed

The full explanation, diagrams and evidence are in
[html/docs/slab_fixes.html](html/docs/slab_fixes.html) and
[html/docs/validation.html](html/docs/validation.html). In one line each:

| # | change | kind | evidence |
|---|---|---|---|
| A1 | step 1: contours as "depth below the nearby trench", the trench as 0 km, every 5 km; checks; `--clip` | rebuilt | synthetic slab < 0.3 km; 1-3 km from PTHA18's contours |
| A2 | step 2: rptha's `nls.lm` optimiser reproduced (MINPACK, 50 iterations, Inf, R's retry) | fidelity | lines within 0.00 m of rptha's R output (puysegur2, makran2) |
| A3 | `optimal` keeps rptha's mesh, repairs only a broken one | rebuilt | PTHA18's contours give PTHA18's mesh on 14/14 zones |
| A4 | at least 2 rows down dip (ReportPTHA p.14) | fidelity | |
| A5 | cell properties as rptha (ellipsoidal midpoint for strike) | fidelity | equal to PTHA18's table to 4e-10 (9 zones) |
| A6 | step 2's mesh saved with a fingerprint, reused by steps 7 and 9 | new | |
| A7 | deprecated `mid` method as rptha has it | fidelity | |
| A8 | clean ends: crooked (> 45 deg off square) or tapering ends cut along a down-dip line, as PTHA18 did by hand; clean ends kept | new | kermadectonga2's cuts within 10 km of PTHA18's hand-made ends; every end of the 8 examples within PTHA18's range |
| B1 | **bug:** the dip correction was applied twice (step 3 and the engine) | bug | +0.6 % to +24 % on the slip rate |
| B2 | scenario probabilities weighted by Bird convergence, Bird shape as LEVEL 4 target | fidelity | every scenario's rate equal to PTHA18's to 1e-10 or better (v7's model: median 4-26 % off) |
| B3 | Bird matching exactly as R (great-circle midpoint, haversine) | fidelity | PTHA18's convergence to 2e-15 on 7 zones |
| B4 | scenarios listed in rptha's order | fidelity | event table row-identical to PTHA18's (4 zones) |
| B5 | LEVEL 4 fit with R's `optimize` tolerance | fidelity | 1e-6 to 1e-12 per scenario |
| B6 | GCMT selection with `gcmt_subsetter.R`'s exact rule | fidelity | PTHA18's event list on 8/8 zones |
| B7 | scenario magnitudes 7.2 to 9.8, as PTHA18 | fidelity | |
| B8 | **bug:** the engine's convergence-profile option crashed | bug | |
| C1-C8 | step 8 on the same engine as step 7; per-folder run names; step 6 reads step 2's settings; ordered logs; tests of the right code; `validate_v9.py`; corrected statements; removed code | plumbing | |

### v8.1

| # | change | kind | evidence |
|---|---|---|---|
| D1 | step 3 checks whether the mesh runs past its own plate boundary, and reports it in the log and the report; `--auto-clip` on step 1 / step_total cuts the flagged columns using a window derived from the mesh (no lon/lat to supply) | new | kurilsjapan: flags 5 of 65 columns, and `--auto-clip` gives 244 cells (PTHA18: 244) and 87.56 mm/yr (PTHA18 recovered: 87.72, was 86.18). Silent on 15 of 17 zones; also flags newguinea's last 2 columns (Bismarck Sea) |
| D5 | step 8 also saves the Bird match **per along-strike column** on PTHA18's mesh (`official_convergence_per_column.json`), so step 9 can export `convergence_profile_official.csv` beside this run's; the rate-curve exports are named `posterior_mean_rate_curve*.csv` | new | official profile: 61 columns, area-weighted mean 87.7244615793 mm/yr = PTHA18's own value |
| D2 | the official mesh in the comparison figures is drawn from PTHA18's own published cell polygons (`validation/ptha18_reference/unit_source_grid/<zone>.shp`) instead of rectangles rebuilt from centroid+length+width+strike | new | cell centroids equal to the netCDF table's to 0.0000 deg on kurilsjapan, kermadectonga2, puysegur2 |

**D1, what it does NOT do.** It only reports. Which columns really belong to a
zone is a geological judgement, so nothing is trimmed unless you ask with
`--auto-clip`. And `--auto-clip` refuses when a lon/lat box cannot separate
the flagged tail from the rest of the arc (newguinea runs east-west, so its
eastern tail sits inside the bounding box of the part you keep); there it
asks for an explicit `--clip` instead of producing a useless one. Because
`--auto-clip` rebuilds the mesh, step_total also passes `--force` to step 6,
which **discards any hand edit** to the input JSON.

| D3 | the depth-along-the-arc profile is drawn on every run, not only when an official mesh exists | new | `figures/mesh_profile.png`, one line per down-dip row |
| D4 | step 9 writes `outputs/exports/*.csv`: the mesh (lon, lat, depth per cell), the posterior mean rate curve and the convergence, each with an `_official` twin when step 8 ran | new | official mesh CSV equals the netCDF table to 0.000000 deg; `convergence_official.csv` = 87.7244615793 mm/yr, PTHA18's own value |

**D1 correction (the directional margin).** The first version padded the clip
box by 0.35 deg on all four sides, including the side the tail was cut from,
which let step 1 re-contour one extra along-strike column back into the mesh
(kurilsjapan: a cell at lat 34.078, still reading Izu-Bonin's 60.2 mm/yr, and
still visible as a dark cell at the tip). The margin is now applied only on
the sides the tail does NOT lie beyond. After the fix kurilsjapan's clipped
mesh reaches lat 34.24 (PTHA18 cuts at 34.30) with a minimum column
convergence of 75.7 mm/yr, i.e. no Izu-Bonin values anywhere.

**D2, the ordering trap.** The netCDF statistics table is down-dip major
((1,1), (2,1), (3,1), (4,1), (1,2)...) while the shapefile is along-strike
major ((1,1), (1,2), (1,3)...). Taking the shapefile's rows in file order
would pair every polygon with another cell's depth and convergence -- wrong,
but plausible-looking. The polygons are therefore indexed by
(downdip_number, alongstrike_number) and read out in the netCDF's order. A
zone with no shapefile falls back to the old rectangles.

### v8.2

| # | change | kind | evidence |
|---|---|---|---|
| E1 | `generate.py --columns trench\|average` (default `trench`, rptha's and PTHA18's rule). `average` counts the columns from the average cell row instead of the trench row; only the column count changes (rptha's down-dip lines, equal-width rows and rectangular grid stay) | new, opt-in | calabria2: 20 x 5 -> 12 x 5, cells with both sides 30-70 km 26% -> 42%, mean deviation 1.05 -> 0.69, rates -0.7% (Mw 7.2+) to -3.7% (Mw 9.2+), Mw max unchanged; antilles2 (`--clip`): 24 x 6 -> 18 x 6, 75% -> 93%, rates +0.3% |
| E2 | step 2 prints the mesh shape next to PTHA18's range, and suggests `--columns average` when the taper is outside it | new | prints only |
| E3 | the epistemic percentiles (LEVEL 5) are computed every 0.1 of Mw from 7.2 to 9.8 (27 magnitudes, steps 6 and 8) instead of at 7.2, 7.6, ..., 9.2; the report's band follows them up to where the 84th percentile reaches zero | new | calabria2: the 6 old values unchanged, whole engine run 26 s; the band used to stop at 9.2 |

**Why `average` exists, and why it is not the default.** Every row of an
rptha mesh has the same number of cells. On a tight arc the deep rows are
much shorter than the trench (calabria2: trench 980 km, 60 km contour
243 km), so with the trench rule the trench cells are about 50 km long and
the deep ones about 12 km. Counting from the average row puts the ~50 km
cells in the middle rows instead (trench cells ~75 km, deep ~24 km): the
taper cannot be removed, only centred. It is not the default because PTHA18
counts from the trench: fed PTHA18's own contours, the trench rule
reproduces PTHA18's column counts (14/14 zones), and `average` would move
away from them (kermadectonga2 73 -> 69, PTHA18 72).

**What it cannot fix.** A zone whose down-dip width changes a lot along
strike keeps uneven cells whatever the rule: caribbean2's fault is 97 km
wide off Puerto Rico and 420 km in the southern Lesser Antilles, and every
column must keep the same rows (about 1,300 combinations of column count,
row count and row placement were tried; none improves it). Step 2 says so
when the width ratio is outside PTHA18's range.

**Unchanged by default.** With `--columns trench` (the default) every mesh,
fingerprint and result is bit-identical to v8.1: the 19 existing example
meshes re-meshed with the new code are byte-identical, `validate_v9.py mesh`
still reproduces PTHA18's 14 published meshes, and example folders generated
before v8.2 (whose steps have no `COLUMN_RULE`) keep rptha's rule. Test:
`pyptha_v12/tests/test_column_rule.py`. Docs:
[html/docs/step2.html#column-rule](html/docs/step2.html#column-rule).

---

## 8. The ten steps, one by one

Run each from `ptha18_logic_tree_test/` as
`.venv/Scripts/python.exe <folder>/steps/<script>`. Every script has a long
docstring at the top explaining its method and sources in full.

### step1_fetch_slab2.py: fault geometry (contours)

- Finds the SLAB grid locally, or downloads it (SLAB2.0 from ScienceBase, SLAB1.0 from USGS).
  Both are plain, unauthenticated downloads. ScienceBase sits behind Cloudflare, which
  sometimes blocks scripted requests with an HTTP 403 "Just a moment..." page. When that
  happens, get the file by hand in a normal browser:
    - SLAB2.0 (zone names ending in `2`): `https://www.sciencebase.gov/catalog/items?q=Slab2+<region title>`,
      open the item titled "Slab2 - ...", download its archive, and extract the single
      `<prefix>_slab2_dep_*.grd` member into `from_scratch_v12/data/slab2/`.
    - SLAB1.0 (zone names without a `2`): `https://earthquake.usgs.gov/static/lfs/data/slab/models/<prefix>_slab1.0_clip.grd`,
      saved as `from_scratch_v12/data/slab1/<prefix>_slab1.0_clip.grd` (exact filename required).
  Each generated `step1_fetch_slab2.py` has these links pre-filled for its region.
- Reads the continuous depth **raster**, not SLAB's contour shapefile.
- **Finds the trench** on the raster (its shallow edge) and measures every depth **below the
  nearby trench**, as PTHA18 does (v8, `lib/slab_contours.py`).
- v10_q, **trench ramp rule** (`TRENCH_RAMP_RISE_KM = 3.0`, `--trench-ramp on`, the default): a
  trench end that climbs in depth like a ramp, more than 3 km deeper than the trench's normal
  depth, is the edge of the SLAB data and is trimmed (at most 100 km) before the datum is built.
  The log prints `ramp rule (...): start trimmed by N km, end by M km`, and
  `data/step1_contours_info.json` keeps `trench_ramp_rise_km` and `ramp_trim_km`. See
  [Trench ramp rule](#trench-ramp-rule-v10_q).
- Draws the trench as the 0 km contour and one contour every 5 km down to the Berryman cutoff.
- **Cleans the two ends** (v8): where the SLAB data (or `--clip`) end the zone crookedly, so
  that the mesh's end cells would be more than 45 deg off square, or where the zone narrows
  to a point, every contour is cut along one down-dip line (a line of steepest descent,
  perpendicular to every contour), as PTHA18 did by hand. A clean end is kept as it is. The
  log says, for each end, what it decided and why.
- **Checks** the contours (a missing level, a level much shorter than the one above it, a
  closed loop, a contour along the grid edge, a broken level) and stops with a message rather
  than meshing a bad set.
- Unwraps the antimeridian (longitudes to [0, 360)) for zones crossing 180 deg.
- Writes `inputs/geometry/<zone>_slab2_contours.shp`, `data/step1_contours_info.json` (with
  each end's decision) and `figures/step1_trench_and_contours.png` (the trench, the contours,
  the end cuts in orange and, dotted, what they removed: look at it once per zone).

### step2_build_grid.py: unit sources

- Cuts the contours into unit sources with the chosen `--discretizer`; the number of columns
  follows `--columns` (`trench` by default, rptha's rule).
- v12: with `--cell-size strasser` the rows and columns are instead raised until no cell is
  longer or wider than the Mmin rupture / `--cell-k` (step 2 prints each try: cells, longest,
  widest, caps). See [`--cell-size strasser`](#--cell-size-strasser-2026-10-07).
- Prints the **mesh shape** next to PTHA18's 43 published meshes (v8.2): taper (trench-row /
  deepest-row cell length; PTHA18 at most 1.17), width ratio (widest / narrowest column;
  at most 3.26), share of cells with both sides 30-70 km, mean deviation from 50 x 50
  (median 0.21). Prints only: no mesh changes because of it.
- Along-strike target 50 km; down-dip width as in section 6; at least 2 rows (`MIN_DOWNDIP_ROWS`).
- Prints the grid size, the mesh check (folded, collapsed, overlapping cells), per-cell
  ranges, and **the two numbers that drive everything: total area and mean dip**
  (angle-weighted, PTHA18's convention), plus the official values when available.
- Writes `data/slab2/unit_source_grid.npy` with its fingerprint, and the mesh figures.

### step3_convergence.py: tectonic convergence

- Uses Bird (2003) plate-boundary velocities (shipped in `data/bird/`, no download).
- PTHA18's exact matching (`lib/bird_convergence.py`): each column's top cell is moved to the
  trench, the nearest non-normal Bird segment is found by great-circle distance, the
  convergent component and the capped lateral component give the convergent slip, and each
  column's value applies to all its cells.
- Area-weights them into the zone's convergence: the **horizontal** plate rate. The engine
  divides it by cos(mean dip) once (v3-v7 divided twice, see 7e B1).
- Writes `data/convergence.txt` and `data/convergence_per_column.json` (used by step 6).
- `CONVERGENCE_OVERRIDE_MM_PER_YR` (top of the script) switches to a constant convergence
  and PTHA18's constant-convergence model (what PTHA18 did for puysegur, 35 mm/yr).
- **Checks whether the mesh runs past its own plate boundary** (v8.1). The Bird match never
  fails, so a mesh that extends beyond its own subduction zone silently reads the *next*
  boundary's convergence at a perfectly normal match distance. Step 3 flags a jump of
  >= 25 mm/yr in `div` between along-strike neighbours that is sustained over >= 2 columns
  at an END of the zone, prints it, and writes `data/plate_boundary_change.json` (deleted
  when there is nothing to report). It **never trims anything**: the report shows the
  warning and names the `--auto-clip` command. Flags kurilsjapan (5 of 65 columns, the
  Izu-Bonin overlap) and newguinea (2 of 23, the Bismarck Sea); silent on the other 15.
  Details: [html/docs/step3.html](html/docs/step3.html).

### step4_fetch_gcmt.py: earthquake catalogue (raw)

- Downloads the Global CMT NDK files (cached in `data/gcmt/`) and parses them into one row per event.
  Monthly files not yet published show as "skipped (HTTP Error 404)": harmless.
- No filtering here.

### step5_subset_gcmt.py: earthquakes on this interface

- PTHA18's `gcmt_subsetter.R` rule, exactly: hypocentre OR centroid within 0.4 deg of a unit
  source (each cell buffered, then joined), depth <= 71 km, Mw >= 7.15, and on one nodal plane
  rake within 50 deg of pure thrust and strike within 50 deg of the unit source nearest the
  centroid. Window 1976-01-01 to 2017-03-01 (PTHA18's, 41.16 years).
- Writes `data/gcmt/<zone>_gcmt_subset.csv`.
- On PTHA18's own mesh this gives PTHA18's own event list (8 of 8 zones); on a from-scratch
  mesh the count can differ slightly because the mesh does.

### step6_write_input.py: the logic-tree input JSON

- Collects geometry + convergence (with its per-column profile) + GCMT subset, and adds
  Berryman's coupling, b and Mw_max_observed.
- Writes `inputs/input_<zone>_scratch.json`, with a `_provenance` note for every number.
- v10: writes `events.rupture_size` (`RUPTURE_SIZE`, from `--rupture-size`): `"rptha"` or `"local"`.
  An input without the key is read as `"rptha"`.
- **Refuses to overwrite an existing JSON**, so your hand edits are safe. Use `--force` to regenerate.

### step7_run.py: the full logic tree

- Runs `from_scratch_v12/python_logic_tree_v12/run_logic_tree.py` on the JSON (LEVELs 0-5, section 4).
- v10 / v10_q: with `events.rupture_size: "local"` the ruptures are built cell by cell
  (`events._local_blocks`) and every conditional-probability weight is multiplied by q
  (`events.coverage_weights`); the log prints `n_events = N (rupture size: local)` and
  `v10_q: conditional probabilities x q (... q from A to B)`. The rate of each magnitude
  (rate_curves, exceedance tables, logic-tree branches) is identical to rptha's rule; what
  changes is each scenario's area, slip and rate, the conditional probabilities, the
  integrated slip and the fitted edge multiplier.
- The engine writes to `runs/python/<zone>_scratch_<folder>/` (a fixed location); step 7 copies everything into `<folder>/outputs/` and deletes that staging folder:
  - `rate_curves.csv`: mean exceedance rate vs Mw
  - `exceedance_rate_percentiles.csv`: mean + percentile curves
  - `logic_tree_branches_<zone>.csv`: every branch with prior and posterior weight
  - `logic_tree_summary.csv`: per-level numbers (area, dip, convergence, edge multiplier, ...)
  - `scenario_rates_<zone>.csv` (rows in PTHA18's event order), `conditional_prob_<zone>.csv`, `integrated_slip_<zone>.csv`, `segmentation.csv`
  - segmented runs (v9): the same four files per segment (`<zone>_<segment>`, all on the zone's scenario table), and `scenario_rates_<zone>_source_zone.csv`, every scenario's rate for the zone as a whole (0.5 x unsegmented + 0.5 x each segment, PTHA18's partial-segmentation percentiles)
  - `outputs/figures/*.png`

### step7b_stochastic_slip.py: heterogeneous (HS) and variable-area-uniform (VAUS) slip variants (optional)

- **Purely additive**: never changes any number step 7 wrote. LEVELs 0-5 use FAUS (Fixed Area Uniform Slip) ruptures only, as PTHA18's published rates do.
- For every FAUS placement at every magnitude, generates rptha's 15 HS realisations per placement (floored at 200 per magnitude) with the ported SFFM generator, and derives each VAUS field from its HS field.
- v10: the FAUS "parent" ruptures follow `events.rupture_size` (the same call step 7 makes), so with `local` the parents, and the window where each HS field's peak can sit, are the cell-by-cell ruptures. The SFFM itself is unchanged and still assumes equal cells inside a footprint (see [Known limitations](#known-limitations-v10_q)). In PTHA18 an HS or VAUS event takes its rate from its parent (`compute_rates_all_sources.R` 1551-1600).
- v10_q (2026-10-06): **HS and VAUS rates, as PTHA18 computes them** (`compute_rates_all_sources.R` 1551-1600, fixed shear modulus; `pyptha_v12/hs_vaus_rates.py`). Each field takes a share of the rate of the FAUS rupture it was drawn from (step 7's `scenario_rates_<zone>.csv`, so run step 7 first): a field with peak slip above 7.5 x the scaling relation's mean slip for its Mw gets rate 0 (`config_peak_slip_limit_factor.R`); the others are weighted by the rank of their peak slip in the family, q = rank/(n+1), with PTHA18's DART-calibrated weights (one curve for HS, one for VAUS), and the shares of one FAUS rupture add up to 1. Both `summary.csv` files gain `faus_event_id`, `above_peak_slip_limit`, `weight_in_family` and `rate_mean`, `rate_p0.025` ... `rate_p0.975` (the parent's times the weight). PTHA18 never published its weight table (`peak_slip_quantile_adjustment_factors.csv`); `data/ptha18_peak_slip_quantile_weights.csv` was recovered from its published HS/VAUS rates by `validation/recover_peak_slip_weights.py`: fitted on puysegur2 alone it reproduces every kermadectonga2 HS and VAUS rate to 1e-14, and with the file every published rate of both zones (101,132 scenarios, every zero rate included) is reproduced to 4e-12. PTHA18's variable shear modulus rates (`variable_mu_*`) are not ported. These rates enter nothing in LEVELs 0-5. calabria2_v10_q: 774 of 16,322 HS and 41 VAUS fields are above the limit; no FAUS rupture is left without a field under it.
- If PTHA18's HS/VAUS files for this zone are available, compares them statistically (not field by field: SFFM is random), see `lib/hs_official_compare.py`.
- Uncapped by default (tens of thousands of fields on a real zone). `--max-placements-per-mw` / `--max-fields-per-mw` cap it. Skip entirely with `step_total.py --skip-hs`.

### step7c_variable_mu.py: variable shear modulus (v11, optional)

- Needs step 7 (`scenario_rates_<zone>.csv`) and step 7b (the HS scenarios).
- Writes `inputs/input_<zone>_scratch_variable_mu.json` (step 6's input plus a
  `variable_shear_modulus` block) and runs the engine on it; checks that the
  engine's constant-rigidity scenario rates equal step 7's before copying
  anything.
- Copies the variable shear modulus files into `outputs/` and adds the
  variable columns to both HS/VAUS `summary.csv` (see
  [What v11 changes](#what-v11-changes)). Nothing step 7 or 7b wrote changes.
- Normal-fault zones keep the constant rigidity, as in PTHA18.

### step8_official.py: official PTHA18 run (optional)

- Needs **R with the rptha package**. Downloads PTHA18's saved session (1.34 GB, cached once) and extracts the official logic tree via `official_ptha_data/fetch_official_inputs.py`.
- Runs it through **the same engine as step 7**, after adding PTHA18's Bird model (per-column Bird values on PTHA18's own mesh) for zones where PTHA18 used Bird. It checks that those values reproduce the official convergence.
- Writes `inputs/input_<zone>_official.json` and `outputs_official/`.
- v9: on a segmented example (step 6 wrote segments) the official run is **PTHA18's own segmented model** (its segment rows and per-segment GCMT events), written to `runs/python/<zone>_official_segmented/`, with the segments listed in `outputs_official/ptha18_reference/official_segments.json`. A zone PTHA18 does not segment stays unsegmented.
- v11: a zone that is not one of PTHA18's 51 (its `sourcezone_parameters.csv`; e.g. calabria2, antilles2) is recognised before anything runs: step 8 writes `outputs_official_status.txt` and exits cleanly, and step 9 then says the zone has no official counterpart instead of asking for step 8. Steps 1-7 and 9 are complete without it.
- v11: works from `V9/`: PTHA18's extracted inputs, trees, public files, `rptha/` and `slab_vs_ptha/` are looked for next to the package and one level up (step 8, `lib/official_geometry_params.py`, `lib/bird_convergence.py`, `lib/plot_meshes.py`); the official input's unit-source table is passed to the engine as an absolute path. Checked: kermadectonga2_v11 and kurilsjapan_v11's official runs are byte-identical to the v9 folders' (validated against PTHA18's published rates).

### step9_report.py: HTML report

- Reads everything back from disk and writes `<folder>/report.html`: what each step did, the real numbers, mesh maps, a rupture explorer, and the from-scratch vs official comparison (if step 8 ran), plus step 7b's HS/VAUS section if it ran.
- v10 / v10_q: the rupture explorer shows each rupture's own block size (with `local` it changes from one placement to the next), and the rupture section says when `--rupture-size local` (and q) was used.
- v10_q (2026-10-06): the rupture explorer has a **Show** switch, FAUS, VAUS or FAUS + VAUS (VAUS needs step 7b's `outputs/vaus_slip_fields/summary.csv`; without it only FAUS can be picked). The FAUS slider and play button step through the FAUS ruptures; a dropdown (1, 2, 3, ...) picks which of that rupture's VAUS realisations is drawn. Colours: FAUS blue, VAUS amber, cells in both striped. Each line gives the block size in cells, area in km2 (and its ratio to the scaling relation's area) and uniform slip. A panel next to the map plots area / scaling-relation area on a log axis (0.25 to 4) with the relation's +-1 sigma range shaded: every FAUS rupture of the Mw and every VAUS realisation of the current rupture as small dots, the ones on the map as large dots with their value (a **Panel** menu switches it between area and slip; hollow dots are VAUS above PTHA18's peak-slip limit, rate 0; each VAUS line gives its share of the FAUS rupture's rate). The text under it explains the scatter (the relation gives a median area; +-1 sigma is about a factor of 2, so about 1 in 3 earthquakes falls outside). A second panel, **Uniform slip, in metres**, shows the same dots as slip (log axis), the slip on the median area (dashed) and PTHA18's HS/VAUS peak-slip limit, 7.5 x that slip (red): PTHA18 gives an HS or VAUS scenario above it zero rate (rptha `EVENT_RATES/config_peak_slip_limit_factor.R`, `compute_rates_all_sources.R` 1569-1590). Since 2026-10-06 step 7b applies it too (see step 7b below). VAUS fields outside +-1 sigma are expected: PTHA18's own VAUS catalogue for kermadectonga2 has 22% outside (p05-p95 about 0.36-2.4 x the median area), calabria2_v10_q 19-31% at Mw 7.2-8.6; at the largest Mw the zone is too small and every field falls below 1. VAUS fields are matched to the FAUS rupture they were drawn from by step 7b's `placement` number (also with `--max-placements-per-mw`).
- v9, segmented runs: a LEVEL 0 section with a bar of how each model divides the trench (this run and PTHA18), a map of the cells coloured by segment (`figures/segments_map.png`), a table per segmentation (columns, convergence, coupling, Mw max observed, GCMT events, rate), this run against PTHA18 for the unsegmented branch, the union and the mix, and a chart of all of them. The rate chart gains each segment, the union and the mix, and its percentile band and the full rate table become the whole zone's (the mix); the step cards, TVD, edge fit and "this run vs PTHA18" table stay the unsegmented branch's, the last one labelled so. Block by block, with screenshots: [html/docs/report_guide.html#modes](html/docs/report_guide.html#modes).
- v9, segmented runs: also `<folder>/report_unsegmented.html`, the unsegmented branch alone, as an unsegmented run's report shows it (built from the same outputs, nothing is run twice). A switch at the top of both pages goes from one to the other.
- The "this run vs PTHA18" box names the input that most likely explains a rate gap; when the Mw 7.2 rates agree within 1% it says so instead of attributing a gap.
- Draws the mesh, and PTHA18's published mesh when it can be downloaded, on a zoomable map over Esri's ocean basemap, to check by eye that the mesh sits on the real trench. This map is the one part of the report that needs an internet connection. All example reports have it (see [html/docs/report_guide.html](html/docs/report_guide.html)).
- The official mesh in the comparison figures is drawn from PTHA18's **own published cell
  outlines** (`validation/ptha18_reference/unit_source_grid/<zone>.shp`), not from
  rectangles rebuilt out of centre+length+width+strike. The two files store their cells in
  different orders (netCDF down-dip major, shapefile along-strike major), so the polygons
  are re-indexed by `(downdip_number, alongstrike_number)` before being paired with depth
  or convergence. A zone with no shapefile falls back to the rectangles.
- `figures/mesh_profile.png` (depth along the arc, one line per down-dip row) is drawn on
  **every** run, with or without step 8.
- Writes `outputs/exports/*.csv`, the numbers every figure is drawn from:
  `mesh_cells.csv` (`downdip_number, alongstrike_number, lon, lat, depth_km,
  convergent_slip_mm_per_yr`, one row per unit source),
  `posterior_mean_rate_curve.csv`, `convergence.csv` and
  `convergence_profile.csv`. Where step 8 supplied PTHA18's own version, an
  `_official` twin with identical columns is written beside it, so all four
  comparisons are available. Segmented runs add a `segment` column to
  `mesh_cells.csv` and `convergence_profile.csv`, the segment, union and
  source-zone curves to `posterior_mean_rate_curve.csv`, and
  `scenario_rates_source_zone.csv` (+ `_official`).
  - **"posterior" is not decoration.** The curve is the logic-tree mean taken with the
    weights *after* the LEVEL 3 Bayesian update against the observed GCMT events
    (`update_logic_tree_weights_with_data`). The official curve is posterior in the same
    sense: step 8 runs PTHA18's own input through this same engine. The report's
    prior-vs-posterior TVD shows how far the update moved the weights.
  - **The two convergence profiles do not line up row by row.** Each has one row per
    along-strike column of *its own* mesh, and the two meshes rarely have the same column
    count (kurilsjapan: 60 here against PTHA18's 61). Compare them as two profiles along
    the arc. Every other `_official` pair *can* be read row by row.

### step_total.py: all of the above

```
.venv/Scripts/python.exe <folder>/steps/step_total.py                              # steps 1-9 + 7b
.venv/Scripts/python.exe <folder>/steps/step_total.py --skip-official              # steps 1-7 + 7b + 9
.venv/Scripts/python.exe <folder>/steps/step_total.py --skip-hs                    # steps 1-9, no HS/VAUS (faster)
.venv/Scripts/python.exe <folder>/steps/step_total.py --skip-official --skip-hs    # LEVELs 0-5 only, fastest
.venv/Scripts/python.exe <folder>/steps/step_total.py --auto-clip                  # cut the columns step 3 flagged
```

- Stops at the first failing **required** step. Steps 7b and 8 are optional: if they fail, it says why and continues.
- Re-running is safe: downloads are cached, and step 6 keeps your edited JSON.

---

## 9. Recipes: commands for every zone

All with the recommended `--ptha false --discretizer optimal`. The folder
name is your choice.

**The eight examples, `*_v9`** (unsegmented; each then run with
`step_total.py --skip-hs`, kurilsjapan with steps 1-3 first and
`step_total.py --skip-hs --auto-clip`):

```
# SLAB2.0 zones
.venv/Scripts/python.exe from_scratch_v12/generate.py kermadec   --zone kermadectonga2 --folder kermadectonga2_v9 --ptha false --discretizer optimal
.venv/Scripts/python.exe from_scratch_v12/generate.py puysegur   --zone puysegur2      --folder puysegur2_v9      --ptha false --discretizer optimal
.venv/Scripts/python.exe from_scratch_v12/generate.py makran     --zone makran2        --folder makran2_v9        --ptha false --discretizer optimal
.venv/Scripts/python.exe from_scratch_v12/generate.py calabria   --zone calabria2      --folder calabria2_v9      --ptha false --discretizer optimal
.venv/Scripts/python.exe from_scratch_v12/generate.py caribbean  --zone antilles2      --folder caribbean2_v9     --ptha false --discretizer optimal
.venv/Scripts/python.exe from_scratch_v12/generate.py caribbean  --zone antilles2      --folder antilles2_v9      --ptha false --discretizer optimal --clip 297,307,5,18.5

# SLAB1.0 zones
.venv/Scripts/python.exe from_scratch_v12/generate.py kuril      --zone kurilsjapan    --folder kurilsjapan_v9    --ptha false --discretizer optimal
.venv/Scripts/python.exe from_scratch_v12/generate.py "south america" --zone southamerica --folder southamerica_v9 --ptha false --discretizer optimal
```

Segmented (v9, LEVEL 0): add `--segmented true` and a folder name of your
choice (`_v9seg` below). Only zones Berryman et al. (2015) divide into
segments (kermadectonga2, kurilsjapan, izumariana, ryuku, sunda2,
southamerica, alaskaaleutians, newhebrides2, solomon2, mexico, newguinea2,
manus); kurilsjapan needs steps 1-3 then `--auto-clip`:

```
.venv/Scripts/python.exe from_scratch_v12/generate.py kermadec --zone kermadectonga2 --folder kermadectonga2_v9seg --ptha false --segmented true
.venv/Scripts/python.exe kermadectonga2_v9seg/steps/step_total.py --skip-hs
.venv/Scripts/python.exe from_scratch_v12/generate.py kuril --zone kurilsjapan --folder kurilsjapan_v9seg --ptha false --segmented true
.venv/Scripts/python.exe kurilsjapan_v9seg/steps/step1_fetch_slab2.py
.venv/Scripts/python.exe kurilsjapan_v9seg/steps/step2_build_grid.py
.venv/Scripts/python.exe kurilsjapan_v9seg/steps/step3_convergence.py
.venv/Scripts/python.exe kurilsjapan_v9seg/steps/step_total.py --skip-hs --auto-clip
```

**Other zones that resolve** (not re-run end to end with v8):

```
.venv/Scripts/python.exe from_scratch_v12/generate.py "new guinea" --zone newguinea2   --folder newguinea2_v9      --ptha false --discretizer optimal
.venv/Scripts/python.exe from_scratch_v12/generate.py solomon     --zone solomon2      --folder solomon2_v9        --ptha false --discretizer optimal
.venv/Scripts/python.exe from_scratch_v12/generate.py cascadia    --zone cascadia      --folder cascadia_v9        --ptha false --discretizer optimal
.venv/Scripts/python.exe from_scratch_v12/generate.py alaska      --zone alaskaaleutians --folder alaskaaleutians_v9 --ptha false --discretizer optimal
.venv/Scripts/python.exe from_scratch_v12/generate.py philippines --zone philippine    --folder philippine_v9      --ptha false --discretizer optimal
.venv/Scripts/python.exe from_scratch_v12/generate.py izu-bonin   --zone izumariana    --folder izumariana_v9      --ptha false --discretizer optimal
.venv/Scripts/python.exe from_scratch_v12/generate.py sumatra     --zone sunda2        --folder sunda2_v9          --ptha false --discretizer optimal
.venv/Scripts/python.exe from_scratch_v12/generate.py vanuatu     --zone newhebrides2  --folder newhebrides2_v9    --ptha false --discretizer optimal
```

Then, for any of them:

```
.venv/Scripts/python.exe <folder>/steps/step_total.py --skip-hs
```

**Notes per zone:**

- `izu-bonin` **needs** `--zone izumariana`; the name is not guessed automatically.
- `kermadec` without `--zone` stops with an "ambiguous" error (3 candidate zones). That is intended.
- `caribbean` covers several slabs; for the Lesser Antilles alone use `--clip 297,307,5,18.5`. A window that cuts through a neighbouring slab (for example `283,307,5,18.5`, which keeps a strip of the Puerto Rico slab) is **not** detected: check `figures/step1_trench_and_contours.png`.
- `southamerica`: SLAB1.0 stops near 43 deg S, so the mesh covers less of the trench than PTHA18's zone.
- Any zone Berryman et al. (2015) does not cover stops early with a clear message from `berryman_params.py`. The supported list is `ZONE_DOMINANT_SEGMENT` in [lib/berryman_params.py](lib/berryman_params.py).

**Comparing methods on the same zone** (keeps everything else equal):

```
.venv/Scripts/python.exe from_scratch_v12/generate.py kermadec --zone kermadectonga2 --folder kt2_optimal --ptha false --discretizer optimal
.venv/Scripts/python.exe from_scratch_v12/generate.py kermadec --zone kermadectonga2 --folder kt2_lm      --ptha false --discretizer lm
.venv/Scripts/python.exe from_scratch_v12/generate.py kermadec --zone kermadectonga2 --folder kt2_ptha    --ptha true  --discretizer optimal
```

---

## 10. Customising a run

### 10.1 Edit the physics by hand

After step 6, open `inputs/input_<zone>_scratch.json` and change what you
need (coupling, `b_anchor`, `mw_max_observed`, `tectonic_convergence_mm_per_yr`,
`desired_unit_source_width`, ...). Then rerun from step 7:

```
.venv/Scripts/python.exe <folder>/steps/step7_run.py
.venv/Scripts/python.exe <folder>/steps/step9_report.py
```

Step 6 will **not** overwrite your edits. To throw them away:
`step6_write_input.py --force`. If you change the mesh settings, re-run
steps 2 and 3 as well: the per-column convergence must describe the same
columns (the engine stops with a clear message if the column count differs).

### 10.2 Force the number of down-dip rows

In `<folder>/steps/step2_build_grid.py` set `N_DOWNDIP_OVERRIDE = 3` (for
example), then rerun steps 2, 3, 6 and 7. Step 6 reads that constant straight
from step 2's source, so the two can never disagree.

### 10.3 Change the contour spacing, the window or the seismogenic cutoff

- **Contour spacing:** `CONTOUR_SPACING_KM` near the top of
  `<folder>/steps/step1_fetch_slab2.py` (default 5 km, PTHA18's).
- **Window:** `CLIP_BBOX` in the same file (what `--clip` sets).
- **Cutoff:** `main()` computes it with `berryman_cutoff_km(...)`. To try
  another value, overwrite the `cutoff` variable right after that call, then
  rerun steps 1 onwards.

### 10.4 Change defaults for every future example

Edit the `.tmpl` files in `templates/`. Placeholders (`{{ZONE}}`,
`{{FOLDER}}`, `{{USE_PTHA}}`, `{{DISCRETIZER}}`, `{{CLIP_BBOX}}`, ...) are
filled by `generate.py`, which refuses to write a file with an unresolved
placeholder. Existing example folders are not affected; regenerate them to
pick up the change.

### 10.5 Plot a mesh on its own

```
.venv/Scripts/python.exe from_scratch_v12/lib/plot_meshes.py kermadectonga2 kermadectonga2_v9   # vs official
.venv/Scripts/python.exe from_scratch_v12/lib/plot_meshes.py --scratch-only kermadectonga2_v9   # alone
```

---

## 11. Verification

**Tests** (about 20 s): `pytest from_scratch_v12/pyptha_v12/tests`: **163
passed, 15 skipped** here (the skipped ones need PTHA18 files not present). New in v10 / v10_q:
`test_rupture_size.py` (11 tests: the default is rptha's rule; with `local` every block fits,
every cell is covered at every magnitude, the moment is exact, areas follow Strasser on an
uneven mesh and are never worse than rptha's on a uniform one, the full-length case, and
`coverage_weights` by hand, equal on rptha's interior, balancing dense and sparse parts) and
`test_trench_ramp.py` (4 tests: a synthetic trench that ramps from 20 to 8 km is trimmed, off
is v9, no ramp means nothing trimmed, the datum near the ramp is the real trench's). `tests/test_v8_fidelity.py` checks against PTHA18's own files and
rptha's R output: the optimiser against MINPACK, the down-dip lines against
rptha in R, the mesh against PTHA18's mesh, the cell properties against
PTHA18's table, step 1 on three synthetic plates (contour positions, a
slanted data edge cut square, a tapering tip cut back), the event list
against PTHA18's, the Bird convergence on 7 zones.

**Against PTHA18's published files** (`validation/validate_v9.py`,
downloads the references from NCI once):

| check | what it compares | result |
|---|---|---|
| `mesh` | PTHA18's contours through v8's discretiser vs PTHA18's published mesh | 14 of 14 zones |
| `engine` | the 32,000 logic-tree branches on the official inputs vs PTHA18's | 4 of 4 zones, to 1e-12 |
| `events` | the rate and 5 percentiles of every scenario vs PTHA18's `rate_annual*` | 4 of 4 unsegmented zones, to 1e-10 or better |
| `gcmt` | step 5's rule on PTHA18's mesh vs PTHA18's own event list | 8 of 8 zones |
| `segmentation` | v9: Bird's segments tile each example's mesh; prints them beside PTHA18's | well-formedness (the comparison is informational) |
| `berryman_segments` | v9: Berryman et al. (2015)'s segment end points placed on PTHA18's own mesh vs PTHA18's boundary indices | 9 zones, all 18 shared boundaries within 2 columns (12 identical) |
| `official_segments` | v9: the engine on PTHA18's segmented inputs vs PTHA18's tree of every segment | 7 zones, every segment, to 1e-11 |
| `events_segmented` | v9: every scenario's rate and 5 percentiles on SEGMENTED zones vs PTHA18's published `rate_annual*` | 7 zones: mean within 1e-10, percentiles within 4e-10 |
| `v8parity` | v9: v9's engine on the v8 folder's own input (`<zone>_v8`) vs that folder's outputs | bit-identical |
| `recover_peak_slip_weights.py` | v10_q: PTHA18's HS/VAUS weight table recovered from its published rates; every published HS and VAUS rate from the published FAUS rates | 2 zones, 101,132 scenarios, to 4e-12 (zero rates exact) |
| `validate_v12_mesh.py` | v12: the alaskaaleutians .dat vs PTHA18's unit-source table; the engine on PTHA18's official input with the .dat in place of the table | rows 2-4 to 1e-10; rates 8.5e-12 with PTHA18's 0 km trench, 7e-4 as sent |
| `validate_v11.py` | v11: every published `variable_mu_Mw` (2 zones, FAUS/HS/VAUS); the engine's `variable_mu_rate_annual` and 5 percentiles on PTHA18's puysegur2 input with its HS catalogue; every HS/VAUS rate, both rigidities | 4e-15 / 7e-13 / 4e-12, zero rates exact |
| `unseg_view` | v9: the unsegmented branch of a segmented folder (`<zone>_v9seg`) vs the unsegmented folder (`<zone>_v9`), this run and official: what `report_unsegmented.html` is built on | 2 zones, bit-identical |

```
.venv/Scripts/python.exe -m pytest from_scratch_v12/pyptha_v12/tests -q
.venv/Scripts/python.exe from_scratch_v12/validation/validate_v9.py mesh engine events
.venv/Scripts/python.exe from_scratch_v12/validation/recover_peak_slip_weights.py
.venv/Scripts/python.exe V9/from_scratch_v12/validation/validate_v11.py   # from ptha18_logic_tree_test/
.venv/Scripts/python.exe from_scratch_v12/validation/validate_v9.py gcmt --ndk kermadectonga2_v9/data/gcmt
.venv/Scripts/python.exe from_scratch_v12/validation/validate_v9.py segmentation official_segments events_segmented v8parity unseg_view
# no argument runs every check. official_segments/events_segmented need PTHA18's trees for their 7 zones:
.venv/Scripts/python.exe official_ptha_data/fetch_official_inputs.py kermadectonga2 kurilsjapan izumariana ryuku sunda2 southamerica alaskaaleutians
```

**v10 / v10_q checks** (all on 2026-10-02, numbers in `V9/html/v10q_rationale.html`):
rupture areas within a factor 1.5 of Strasser at Mw 7.2-8.7 on 31 PTHA18 meshes and 4 of ours
(rptha 46-100%, local 85-100%; never a cell left out of every rupture of a magnitude); moment
identity (sum of weight x slip x area = M0 / rigidity at every magnitude, 12 runs, to 1e-14);
exceedance tables and rate curves identical to rptha's rule (to 3e-13); PTHA18's misfit with
v10_q / with rptha's rule: Calabria 0.85, Caribbean 0.70, Kermadec-Tonga 0.72, Kurils-Japan 0.91;
the "classic" control runs reproduce the v9 folders exactly; the ramp rule changes only
hellenic, caribbean/antilles and kermadectonga2's trenches.

**End to end:** the eight examples were run with v9 on 2026-09-30
(`*_v9`: steps 1-9 with the official step 8 where PTHA18 published the zone,
`--skip-hs`): all complete, all meshes with 0 broken cells. Their numbers
are in [html/docs/examples.html](html/docs/examples.html). Two also have a
segmented twin, `kermadectonga2_v9seg` and `kurilsjapan_v9seg`
([html/docs/segmentation.html](html/docs/segmentation.html)). Against the
v8 folders (`*_v8`, made with `from_scratch_v8/generate.py`, step 7b on
puysegur2) steps 1-5 give the same files (kurilsjapan apart, now clipped);
from step 6, kermadectonga2, southamerica and puysegur2 move with v9's zone
Berryman values, and makran2, calabria2, caribbean2 and antilles2 give v8's
numbers.

---

## 12. Known caveats

1. **Segmentation follows Berryman, not PTHA18's own table.** v9 runs LEVEL 0 with `--segmented true`, and the boundaries come from Berryman et al. (2015) Table 3.1's segment end points, the table PTHA18 took its segments from, never from PTHA18's `sourcezone_parameters.csv`. On PTHA18's own meshes they land within 0 to 2 columns of PTHA18's; the counts differ only where PTHA18 merged Berryman's segments (Patagonia, eastern Alaska) or added one (Arakan on sunda2). On this run's own mesh a boundary can move a column or two more, because the mesh differs. The METHOD is PTHA18's and is checked exactly (segment trees and every scenario's rate on 7 segmented zones, from PTHA18's inputs). Without the flag, every zone is modelled unsegmented as in v8.
2. **Puysegur's convergence.** PTHA18 used a fixed 35 mm/yr there, not Bird. The pipeline uses Bird everywhere (about 25 mm/yr on puysegur2). `CONVERGENCE_OVERRIDE_MM_PER_YR = 35` in step 3 reproduces PTHA18's choice.
3. **Cell boundaries follow SLAB**, not PTHA18's hand-edited contours (published on NCI; fed those, v8 gives PTHA18's mesh). Judge a from-scratch mesh on **total area and mean dip**, the two mesh numbers the moment balance uses. Step 1's clean ends only ever **cut**: where PTHA18 extended a zone by hand beyond the SLAB data (puysegur2's southern tip), v8 does not invent that geometry, so it ends a little shorter there.
4. **Coupling, b and Mw_max_observed come from Berryman et al. (2015)**, the paper PTHA18 cites, not from PTHA18's own table; the values differ slightly.
5. **`--clip`** windows that cut through a neighbouring slab are not detected (see section 9).
6. **Bird tables.** `--convergence bird` (default) reads Bird's public catalogue; `--convergence bird-griffin` reads PTHA18's table (Bird plus Jonathan Griffin's traces, a copy in `data/bird/`). They agree where PTHA18 used Bird (kermadectonga2 +0.2%, makran2 -0.1% on these meshes) and differ a lot where it used Griffin's traces (newguinea2: 38 vs 96 mm/yr). Per zone: `html/docs/convergence_sources.html`.
7. **`python_logic_tree_v12/gr_curve_and_convergence.py`** is an unused leftover of the fork; the pipeline never calls it. `cutoff_km` and `PUBLISHED_CUTOFF_KM` in `lib/official_geometry_params.py` are also unused (both already were in v7).
8. **Step 8 needs R + rptha** and a zone PTHA18 modelled; otherwise it is skipped. It writes the official run to the shared `runs/python/<zone>_official/` (the same content whichever example runs it), or, on a segmented example, `runs/python/<zone>_official_segmented/`.
9. **Step 7b's VAUS peak slip runs slightly above PTHA18's published VAUS catalogue** (kermadectonga2: +1% to +14%, growing with Mw), traced to the from-scratch mesh's slightly different cell areas, not to the rule.
10. **v10 / v10_q:** see [Known limitations (v10_q)](#known-limitations-v10_q) (one-cell minimum, the HS generator's equal-cell assumption, cell shape, not PTHA18's procedure with `local`, Berryman's dip).

---

## 13. Glossary

| term | meaning |
|---|---|
| **PTHA18** | Geoscience Australia's 2018 Probabilistic Tsunami Hazard Assessment. This pipeline reproduces its earthquake-source part. |
| **rptha** | the R package PTHA18 was built with. `pyptha_v12` is a Python port of the parts needed here. |
| **NCI** | the server where PTHA18 published its files (thredds.nci.org.au, `fj6/PTHA/AustPTHA_1`). |
| **SLAB2.0 / SLAB1.0** | USGS 3-D models of subducting slabs; the depth raster gives the fault surface. |
| **trench** | where the plate starts to dive; the shallow edge of the fault, and PTHA18's 0 km contour. |
| **depth below the nearby trench** | PTHA18's depth convention: depth of a point minus the depth of the nearest trench point. |
| **contour** | a line of equal depth on the fault surface. |
| **shear modulus, rigidity (mu)** | how stiff the rock is; M0 = mu x area x slip. Constant 30 GPa in steps 7/7b; depth-varying in step 7c (v11). |
| **variable_mu (v11)** | PTHA18's depth-varying rigidity case: each scenario's magnitude relabelled (`variable_mu_Mw`), LEVEL 3 weights with that difference as a catalogue magnitude error, rates `variable_mu_rate_*`. Slip and tsunami unchanged. |
| **unit source** | one roughly 50 x 50 km patch of the fault. Ruptures are built from blocks of them. |
| **discretiser** | the algorithm that cuts the contour surface into unit sources. |
| **down-dip / along-strike** | down the slope of the fault / along the trench. |
| **bow-tie** | a self-intersecting (folded) cell, i.e. a broken mesh. |
| **seismogenic cutoff** | the maximum depth (below the trench) at which the interface produces large earthquakes. |
| **convergence** | how fast the plates move toward each other (mm/yr), the driver of moment accumulation. |
| **coupling** | the fraction of convergence released in earthquakes (the rest creeps). |
| **Gutenberg-Richter (GR), b-value** | the magnitude-frequency law; b sets how fast rates drop with magnitude. |
| **Mw_max** | the largest magnitude allowed on a branch. |
| **moment balance** | long-term earthquake moment = area x slip rate x rigidity; it fixes each branch's rate level. |
| **logic tree / branch** | one combination of uncertain parameters, with a weight. |
| **scenario (event)** | one earthquake of a given magnitude on a given block of unit sources. |
| **conditional probability** | given an earthquake of magnitude Mw, the chance that it is this scenario. |
| **edge multiplier** | LEVEL 4's factor on scenarios touching the zone's ends, fitted to the convergence shape. |
| **rupture size: rptha / local** | v10: how many cells a uniform-slip rupture gets. rptha: one block per magnitude from the zone's mean cell size; local: a block per placement from the real km of its cells. |
| **q** | v10_q: per rupture, the mean over its cells of 1 / (number of same-magnitude ruptures containing that cell); multiplies the conditional-probability weight with `local`, so places covered by more ruptures do not get more rate. |
| **trench ramp rule** | v10_q: a trench end more than 3 km deeper than the trench's normal depth is the edge of the SLAB data, not trench, and is trimmed before the datum is built. |
| **FAUS / HS / VAUS** | fixed-area uniform slip (what LEVELs 0-5 use) / heterogeneous stochastic slip / variable-area uniform slip derived from HS (step 7b). |
| **SFFM** | Synthetic Finite Fault Model, the S_NCF spectral method of Davies et al. (2015) rptha uses to generate HS fields. |
| **GCMT** | Global Centroid Moment Tensor catalogue of earthquakes. |
| **Bird (2003)** | the plate-boundary velocity model used for convergence. |
| **Berryman et al. (2015)** | GEM subduction-interface characterisation, the source for coupling, b, Mw_max_observed and the cutoff. |

---

## 14. Further reading

- [html/docs/index.html](html/docs/index.html): the full documentation, with diagrams and evidence.
- [html/docs/data_provenance.html](html/docs/data_provenance.html): with `--ptha false`, exactly which step reads what, and a from-zero run proving it.
- [validation/validate_v9.py](validation/validate_v9.py): the checks against PTHA18's files.
- The docstring at the top of every `templates/step*.py.tmpl` (or of the generated `steps/step*.py`): the full method and sources for that step.
- [generate.py](generate.py) module docstring: the version lineage.
- `V9/html/v10q_rationale.html`: why v10_q, with Calabria / Caribbean switch and the control zones.
- `V9/html/slip_comparison_v9_v10.html`: slip and sea-floor uplift of every rupture, Calabria v9 / v10 / v10_q.
