# Validating pyptha against rptha (R)

This answers the question: **how do we know the Python port matches the R
original, and what are the inputs / outputs of such a test?**

## The idea: golden-value (characterisation) testing

A port is validated by comparing it to the original on a *fixed, known input*:

```
   fixed input  ──►  R code (rptha)     ──►  output_R   (the "golden value")
   fixed input  ──►  Python code (pyptha) ──►  output_py
   PASS if  output_py ≈ output_R   (within a stated tolerance)
```

So each test needs three things:

| Piece            | Where it comes from                                             |
| ---------------- | -------------------------------------------------------------- |
| **the input**    | a concrete case (a magnitude, a fault geometry, an obs. count) |
| **the golden output** | what R produces for that input                            |
| **the tolerance**| how close is "equal" (numerical round-off, integration error)  |

## Where the golden values come from (no R install needed)

rptha ships its own unit tests in `rptha/R/rptha/tests/testthat/*.R`. Those files
**already contain the exact inputs and the expected numeric outputs** that the
rptha authors assert against — many taken from the original literature. We
reuse them directly, so we get a 1:1 comparison with R **without running R**.

`test_r_parity.py` does exactly this. Highlights:

- **Okada deformation** — reproduces Okada (1985) Table, Case 2 (the same case
  `test_okada_tsunami.R` uses). Golden `(East, North, Up)` displacements:
  - dip-slip:    `( 3.527e-2, -4.682e-3, -3.564e-2 )`
  - strike-slip: `( 4.298e-3, -8.689e-3, -2.747e-3 )`
  - tolerance `1e-5` (same as R). This validates the pure-NumPy Okada rewrite
    against the *original paper*, not just against R.
- **Rupture scaling** — `Mw_2_rupture_size(9.0, 'Strasser')` golden
  `area/width/length = 123595 / 189 / 614` and `log10 sigmas = 0.304 / 0.173 /
  0.18` (from `test_rupture_scaling.R`).
- **Moment ↔ magnitude** — `M0_2_Mw(5.66, inverse) = 3.47e17` (Bird et al. 2009).
- **Geometry** — a planar 2×2 source with a known analytic dip
  (`atan((10-6)/20)`), from `test_discrete_source_summary_statistics.R`.

The other test files (`test_scaling.py`, `test_okada.py`, `test_rates.py`, …)
validate against **physical invariants** — properties the code must satisfy
regardless of implementation. These catch subtle port errors a single happy-
path number might miss:

- `rates`: the seismic-moment balance must recover the input slip rate exactly.
- `events`: every generated rupture must have seismic moment exactly `M0(Mw)`.
- `okada`: a thrust must uplift the hanging wall; superposition must hold.
- `stochastic_slip`: clipping must give min slip 0; mean slip is preserved.

## Running the tests

```bash
.venv/Scripts/python -m pytest pyptha/tests/ -q          # all tests
.venv/Scripts/python pyptha/tests/test_r_parity.py       # just the R-parity set
```

## If you ever want a *live* 1:1 comparison against R

The parity tests above are strong, but they only cover the cases rptha's own
suite covers. To compare on *arbitrary* inputs (e.g. your new source zone),
install R and dump R's outputs to compare against:

1. Install R + Rtools (Windows) and build `rptha`
   (see `R/README.md` and `R/install/`).
2. Write a tiny R script that runs the rptha function on your chosen input and
   writes the result to CSV, e.g.:

   ```r
   library(rptha)
   v <- Mw_2_rupture_size(8.7, relation='Strasser')
   write.csv(as.list(v), 'golden_size_8p7.csv', row.names=FALSE)
   ```
3. Load that CSV in a Python test and assert `pyptha` matches it:

   ```python
   import pandas as pd, numpy as np
   from pyptha import scaling
   golden = pd.read_csv('golden_size_8p7.csv')
   got = scaling.Mw_2_rupture_size(8.7, relation='Strasser')
   np.testing.assert_allclose([got['area'], got['width'], got['length']],
                              golden.iloc[0].values, rtol=1e-6)
   ```

This is only needed if you want to validate inputs beyond the built-in cases;
for the ported functionality, the R-parity + invariant tests already pin the
behaviour to R.

## Tests of what is NOT a port (v10, v10_q)

Some behaviour has no R original to compare against, because rptha does not
have it. Those parts are tested against invariants instead, and the R-parity
tests above check that the default path is still rptha's:

| file | what it tests |
| --- | --- |
| `test_rupture_size.py` | v10 `--rupture-size local` (`events._local_blocks`): the default is rptha's rule (same blocks); with `local` every block fits on the mesh, every cell is in at least one rupture of every magnitude, no duplicate ruptures, the moment of every rupture is exactly M0(Mw); on a calabria2-like uneven mesh the areas follow Strasser (all within x1.6, spread < x2) where rptha's single block is off by more than x2.5; on a uniform mesh local is never worse than rptha; a rupture longer than the zone is the full length with rows by area; the event table passes the rule through. v10_q `events.coverage_weights` (q): a hand-computed case, equal q for every interior rupture of rptha's rule on equal cells, and more even per-cell coverage with q than without. |
| `test_trench_ramp.py` | v10_q trench ramp rule (`lib/slab_contours.find_trench(ramp_rise_km=)`): a synthetic trench that ramps from 20 km to 8 km over 80 km at one end is trimmed there (and only there); `None` gives v9's trench; a trench with no ramp is not trimmed; the depth below the trench of a point near the ramp is measured from the real trench with the rule. |

| `test_hs_vaus_rates.py` | v10_q `hs_vaus_rates.family_weights` (PTHA18's HS/VAUS rates, `compute_rates_all_sources.R` 1551-1600): the weight table (51 knots, integral 1, HS falling and VAUS rising with peak slip); a family sums to 1 and a field above 7.5 x the mean slip gets 0; a family entirely above the limit carries nothing; ties ranked in generation order; and, when `official_ptha_data/public_nc/` is present, every published HS and VAUS rate of puysegur2 and kermadectonga2 from the published FAUS rates, to 1e-9. |

| `test_variable_mu.py` | v11 `variable_mu` (PTHA18's variable shear modulus): the rigidity curve at its knots and its log interpolation; the relabelled magnitude of a shallow and a deep cell and of a slip string; `make_conditional_ecdf` against R's semantics (right-continuous, linear between bins, clamped, evenly spaced bins required); `MwRateFunction.with_mw_error` uses the error weights (mean and quantiles); the four weight curves; and, when PTHA18's files are present, every published `variable_mu_Mw` of puysegur2 and kermadectonga2 VAUS to 1e-9. |

Count on 2026-10-06 (v11): 179 passed, 15 skipped (the skipped tests need PTHA18
files that step 8 extracts).
