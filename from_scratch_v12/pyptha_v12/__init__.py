"""pyptha - a partial Python port of Geoscience Australia's rptha package.

Scope: earthquake rupture generation and individual scenario rates for
subduction source zones. The tsunami-simulation and hazard-curve parts of
rptha are intentionally NOT ported.

Ported so far:
    scaling  - empirical rupture scaling relations (rupture_scaling.R)
    okada    - Okada (1985) rectangular-fault surface deformation
               (okada_tsunami.R + src/okada_tsunami_fortran.f), pure NumPy
    rates    - Gutenberg-Richter logic-tree scenario rates
               (rupture_probabilities.R)
    unit_sources - unit-source grid -> per-subfault summary statistics
               (unit_sources.R)
    events   - uniform-slip rupture generation (rupture_events.R)
    stochastic_slip - heterogeneous-slip SFFM generator
               (sffm_fit_simulate_earthquake.R)
    contour_discretisation - orthogonal contour -> grid discretiser,
               rptha-faithful (downdip_3d_lines_on_source.R)
    logic_tree - the remaining PTHA18 logic-tree levels: segmentation,
               edge correction, and epistemic percentiles across segments
               with a comonotonic copula
    moment_balance - seismic-moment-conservation diagnostics from the PTHA18
               driver scripts: integrated slip per unit source, the numerical
               fit of the edge multiplier, and the coupling prior
               (EVENT_RATES/back_calculate_convergence.R,
                EVENT_RATES/compute_rates_all_sources.R)
"""

from . import (contour_discretisation, events, logic_tree, moment_balance,
               okada, rates, scaling, stochastic_slip, unit_sources)

__all__ = ["scaling", "okada", "rates", "unit_sources", "events",
           "stochastic_slip", "contour_discretisation", "logic_tree",
           "moment_balance"]
__version__ = "0.0.1"
