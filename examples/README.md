# Examples

Eight finished runs of the generator, one folder each. Open
`<folder>/report.html` in a browser for the full report (geometry, plate
convergence, seismicity, logic tree, scenario rates).

| Folder | Zone | Geometry | What it shows |
|---|---|---|---|
| `kermadectonga2` | Kermadec-Tonga | SLAB2.0 | the main running example of the documentation |
| `kurilsjapan` | Kamchatka, Kuril Islands, northern Japan | SLAB | `--auto-clip`: cuts the tail that runs onto the Izu-Bonin trench |
| `alaskaaleutians_mesh` | Alaska-Aleutians | external mesh, `inputs_meshes/alaskaaleutians_quadrilateral_coors.dat` | `--mesh-file`: PTHA18's own 312-cell mesh instead of one built from SLAB |
| `alaskaaleutians_slab` | Alaska-Aleutians | SLAB | the same zone from SLAB, to compare with the external mesh |
| `calabria2` | Calabria | SLAB2.0 | a small mesh whose cells differ a lot in size (`--rupture-size local`) |
| `caribbean2` | Lesser Antilles | SLAB2.0 | `antilles2` zone parameters on the Caribbean SLAB region |
| `makran2` | Makran | SLAB2.0 | |
| `puysegur2` | Puysegur | SLAB2.0 | |

## How they were made

All eight use `--ptha false --rupture-size local` (no PTHA18 file is read, each
rupture is sized from the real km of its cells) and were run with
`step_total.py --skip-official`: every step runs, including step 7b (the HS and
VAUS slip fields), except step 8, the comparison with PTHA18's official run,
which needs R. The exact commands are at the top of each folder's `RUN.html`
("Commands used for this folder"): the `generate.py` line with its flags, and
every `step_total.py` run with the flags it was given. Run from the repository
root:

```bash
python from_scratch_v12/generate.py kermadec  --zone kermadectonga2 --folder examples/kermadectonga2 --ptha false --rupture-size local
python from_scratch_v12/generate.py caribbean --zone antilles2      --folder examples/caribbean2     --ptha false --rupture-size local
python from_scratch_v12/generate.py calabria  --zone calabria2      --folder examples/calabria2      --ptha false --rupture-size local
python from_scratch_v12/generate.py makran    --zone makran2        --folder examples/makran2        --ptha false --rupture-size local
python from_scratch_v12/generate.py puysegur  --zone puysegur2      --folder examples/puysegur2      --ptha false --rupture-size local
python from_scratch_v12/generate.py alaska    --zone alaskaaleutians --folder examples/alaskaaleutians_slab --ptha false --rupture-size local
python from_scratch_v12/generate.py alaska    --zone alaskaaleutians --folder examples/alaskaaleutians_mesh --ptha false --rupture-size local --mesh-file inputs_meshes/alaskaaleutians_quadrilateral_coors.dat
python from_scratch_v12/generate.py kuril     --zone kurilsjapan    --folder examples/kurilsjapan    --ptha false --rupture-size local

# every folder:
python examples/<folder>/steps/step_total.py --skip-official

# kurilsjapan runs twice: the first run lets step 3 find the plate boundary
# change, the second one cuts the tail with --auto-clip
python examples/kurilsjapan/steps/step_total.py --skip-official --auto-clip
```

`--folder` may be nested (here `examples/<name>`); the generated steps find the
code by climbing out of the folder.

The raw GCMT catalogue (`data/gcmt/`, about 25 MB per example) is not kept in the
repository: step 4 downloads it again when you re-run a folder.
