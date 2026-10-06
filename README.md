# ptha-python-from-scratch

A Python generator that builds a complete, self-contained **PTHA18-style
earthquake source-zone example** for any subduction zone covered by USGS
SLAB2.0 / SLAB1.0, from public data only: fault geometry, plate convergence,
seismicity, logic tree, scenario ruptures, scenario rates and an HTML report.

It is a Python port of the rupture and rate parts of
[rptha](https://github.com/GeoscienceAustralia/ptha) (Geoscience Australia),
the R package behind the 2018 Australian Probabilistic Tsunami Hazard
Assessment (PTHA18), and it adds several options on top of it (external
quadrilateral meshes, local rupture sizing, variable shear modulus rates,
segmented zones). The code lives in [`from_scratch_v12/`](from_scratch_v12/);
its own [README](from_scratch_v12/README.md) is the full reference and
[`from_scratch_v12/html/docs/index.html`](from_scratch_v12/html/docs/index.html)
is the illustrated documentation (open it in a browser).

## Quick start

Tested on Windows 11 with Python 3.13. The code uses `os.path` throughout, but
Linux and macOS have not been tried yet.

```bash
git clone <this repository>
cd <this repository>

python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux, macOS: source .venv/bin/activate
pip install -r requirements.txt

# which zones exist
python from_scratch_v12/list_regions.py

# generate one example (a new folder next to from_scratch_v12/) and run it
python from_scratch_v12/generate.py calabria --zone calabria2 --folder calabria2_v12 --ptha false --rupture-size local
python calabria2_v12/steps/step_total.py --skip-hs --skip-official
# then open calabria2_v12/report.html
```

`generate.py` writes the folder in the repository root; the root `.gitignore`
ignores every generated folder. The scripts inside it print commands as
`.venv/Scripts/python.exe ...` (Windows); use `python ...` on other systems.

External mesh instead of SLAB (the example file is PTHA18's own Alaska mesh):

```bash
python from_scratch_v12/generate.py alaska --zone alaskaaleutians --folder alaskaaleutians_v12 --ptha false --rupture-size local --mesh-file alaskaaleutians_quadrilateral_coors.dat
python alaskaaleutians_v12/steps/step_total.py --skip-hs --skip-official
```

## What needs the network

Downloaded on first use: SLAB2.0 rasters (USGS ScienceBase), SLAB1.0 rasters
(USGS), the GCMT catalogue (globalcmt.org). ScienceBase sometimes sits behind a
Cloudflare bot check that no script can pass, and it can last for hours. So the
small SLAB2.0 rasters of eight regions are cached in
`from_scratch_v12/data/slab2/` (Calabria, Caribbean, Hellenic, Makran, Kermadec,
Puysegur, Sumatra-Java, New Guinea). For any other region, if step 1 is
blocked, it prints the link: download the file in a normal browser and put the
`.grd` in `<your folder>/data/slab2/`.

## Checks

```bash
python -m pytest from_scratch_v12/pyptha_v12/tests -q
```

181 tests pass; 21 are skipped because they compare against PTHA18's published
files, which are downloaded by step 8 (see below) and are not in this repository.

Generating `calabria2` as above reproduces the author's reference run: the
`outputs/` CSV and JSON files are byte-identical.

## Step 8: comparison with PTHA18's official run (optional, needs R)

Steps 1 to 7, 7b, 7c and 9 are pure Python. Step 8 compares your run with
PTHA18's own, and PTHA18's logic-tree weights only exist inside R objects, so it
needs R. Install it yourself first:

1. R (and Rtools on Windows, rptha has Fortran code).
2. The `rptha` R package, from a clone of
   [GeoscienceAustralia/ptha](https://github.com/GeoscienceAustralia/ptha) (the
   `rptha/` folder of this repository only holds a few data files, not the
   package). The exact commands are printed by step 8 if `Rscript` is missing.

Then run `step_total.py` without `--skip-official`, or run
`<folder>/steps/step8_official.py` on its own. On first use it downloads, by
itself, into `official_ptha_data/` (git-ignored): PTHA18's saved R session
(1.4 GB, once, from NCI) and the zone's published unit-source table; R then
extracts the official logic tree from the session, which takes minutes. A zone
PTHA18 never modelled has no official run, and step 8 says so. If `Rscript` is
missing, step 8 stops with the instructions above, and `step_total.py` carries
on to the report.

The scripts in `from_scratch_v12/validation/` check the code against PTHA18's
published files and use the same R setup.

## Third-party material

- `rptha/`: the few files of rptha the code opens (the PTHA18 source zone table
  and two test shapefiles), BSD 3-Clause, (c) 2015 Geoscience Australia. See
  [rptha/LICENSE](rptha/LICENSE).
- `from_scratch_v12/data/bird/`: plate boundary tables, see the README in that
  folder for provenance and terms.
- `from_scratch_v12/data/slab2/`: USGS SLAB2.0 depth rasters (public domain).
- `from_scratch_v12/validation/ptha18_reference/`: small tables published with
  PTHA18 by Geoscience Australia.
- `alaskaaleutians_quadrilateral_coors.dat`: Alaska-Aleutians mesh equal to
  PTHA18's own unit-source mesh (see `from_scratch_v12/README.md`, "What v12
  changes").

## License

See [LICENSE](LICENSE).
