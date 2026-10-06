"""Stochastic (heterogeneous) slip fields - the SFFM core.

Python port of the synthetic finite-fault-model (SFFM) generator in
``rptha/R/sffm_fit_simulate_earthquake.R``: given a corner-wavenumber
parameter and a template grid, produce a random slip surface with the target
amplitude spectrum (the S_{NCF} algorithm of Davies et al. 2015). This is the
heterogeneous-slip (HS) counterpart to the uniform-slip ruptures in
:mod:`pyptha.events`.

Ported:
    sffm_get_default_model_parameters   - default spectrum / clipping / decay
    sffm_get_numerical_wavenumbers      - numerical wavenumber grids
    sffm_recentre_slip                  - move the peak to a target location
    sffm_simulate                       - generate one random slip field
    sffm_make_random_lwkc_function      - per-event random L/W/kcx/kcy draw
        (rptha's own regression, Davies et al. 2015): each HS event gets
        its OWN rupture length/width and corner wavenumbers, drawn from a
        log-normal model fitted to real earthquake finite-fault-model
        statistics, not a single fixed value reused for every event at a
        magnitude.
    rectangle_on_grid                   - place an (L x W) footprint on a
        grid, clamped at its boundary, centred as close to a target cell as
        the boundary allows -- rptha's own placement logic, used together
        with the L/W draw above so an HS realisation's footprint is not
        confined to its parent FAUS rupture's own cells.

    Confirmed to matter, in that order: without the random L/W/kcx/kcy draw
    (every realisation reusing one fixed footprint/wavenumber pair), mean
    peak slip came out systematically 10-25% below PTHA18's own published
    catalogue, worsening at higher Mw. Fixing the wavenumber draw alone
    (footprint still fixed to the parent rupture) narrowed but did not
    close that gap; combining it with rectangle_on_grid placement on the
    FULL zone mesh (a large random footprint draw can spill well beyond the
    parent rupture) closed most of the rest; also randomising the
    footprint's CENTRE (rptha's vary_peak_slip_location -- see
    step7b_stochastic_slip.py, which samples it from a window equal to the
    parent rupture's own real alongstrike/downdip index span -- NOT a
    "typical size at this magnitude" formula; that formula exists in
    sffm_fit_simulate_earthquake.R but only as a fallback R's own driver
    never actually takes, since make_all_earthquake_events.R always passes
    the parent rupture's real extent explicitly) closed the remainder, to
    ordinary realisation-to-realisation scatter around the official mean.

Scope note: the ``sub_sample_size`` refinement path within ``sffm_simulate``
itself (an optional resolution-refinement trick, not part of the core
spectral algorithm) is not ported. netCDF table export (rptha's full
wrapper, ``sffm_make_events_on_discretized_source``, writes its output
straight to a netCDF row) is also not reproduced here -- this run keeps
its output as an in-memory Python table/CSV instead. The nearest-unit-
source-from-lon/lat lookup IS reproduced (step7b_stochastic_slip.py's
``_nearest_unit_source_to``, using the same haversine-distance-to-centroid
method as R's ``which.min(distHaversine(...))``), together with the
spherical centroid (``geomean``) it is searched from and the
vary_peak_slip_location window described above -- all three live in
step7b_stochastic_slip.py, which calls ``sffm_make_random_lwkc_function``
and ``rectangle_on_grid`` directly against this run's own mesh. The
spectral algorithm and both pieces of the random-parameter model - the
parts with scientific content - are ported faithfully.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class SFFMParameters:
    """Configuration for the SFFM generator (port of the R default list)."""

    random_phase_generating_field: Callable[[int], np.ndarray] = \
        field(default=lambda n: np.random.standard_normal(n))
    negative_slip_removal_function: Callable[[np.ndarray], np.ndarray] = \
        field(default=lambda x: np.maximum(x, 0.0))
    spatial_slip_decay: str = "gaussian"   # 'gaussian' | 'exponential' | 'none'
    recentre_slip: bool = True

    def spectral_amplitude_fun(self, kx, ky, reg_par):
        """Default spectrum: (1 + ((kx/kcx)^2 + (ky/kcy)^2)^2)^(-1/2)."""
        return (1.0 + ((kx / reg_par[0]) ** 2
                       + (ky / reg_par[1]) ** 2) ** 2) ** (-0.5)


def sffm_get_default_model_parameters():
    """Return a fresh :class:`SFFMParameters` with rptha's defaults."""
    return SFFMParameters()


def sffm_get_numerical_wavenumbers(tg_mat):
    """Numerical wavenumber grids (kx, ky) for a template matrix.

    For an ``N x M`` grid: kx' = min(0..M-1, M-(0..M-1))/M along columns,
    ky' = min(0..N-1, N-(0..N-1))/N along rows. Returns (kx, ky) matrices of
    the same shape as ``tg_mat``.
    """
    tg = np.asarray(tg_mat)
    B1, A1 = tg.shape
    j = np.arange(A1)
    i = np.arange(B1)
    X1 = np.minimum(j, A1 - j) / A1
    X2 = np.minimum(i, B1 - i) / B1
    kx = np.broadcast_to(X1, (B1, A1)).copy()
    ky = np.broadcast_to(X2[:, None], (B1, A1)).copy()
    return kx, ky


def sffm_recentre_slip(m1, tg=None):
    """Shift the slip field so its peak matches a target (or moves inward).

    If ``tg`` is given, cyclically roll rows/cols so the maximum of ``m1``
    lands where ``tg``'s maximum is. Otherwise, put the row/col with the
    smallest max on an edge. Port of ``sffm_recentre_slip``.
    """
    m = np.asarray(m1, dtype=float).copy()
    nr, nc = m.shape

    if tg is None:
        row_to_bottom = int(np.argmin(m.max(axis=1)))  # 0-based
        col_to_left = int(np.argmin(m.max(axis=0)))
        if row_to_bottom not in (0, nr - 1):
            order = np.r_[row_to_bottom:nr, 0:row_to_bottom]
            m = m[order, :]
        if col_to_left not in (0, nc - 1):
            order = np.r_[col_to_left:nc, 0:col_to_left]
            m = m[:, order]
        return m

    tgm = np.asarray(tg, dtype=float)
    new_r, new_c = np.unravel_index(np.argmax(tgm), tgm.shape)
    old_r, old_c = np.unravel_index(np.argmax(m), m.shape)
    m = np.roll(m, shift=(new_r - old_r), axis=0)
    m = np.roll(m, shift=(new_c - old_c), axis=1)
    return m


def sffm_simulate(reg_par, tg_mat, sffm_pars=None, rng=None):
    """Generate one random slip surface with the SFFM spectrum.

    Parameters
    ----------
    reg_par : sequence
        ``(kcxN, kcyN, ...)`` - the numerical corner wavenumbers (physical
        corner wavenumbers times the cell spacings dx, dy).
    tg_mat : 2D array
        Template grid; its shape (and the location of its maximum, used for
        recentring) drive the output. Values give the target total slip.
    sffm_pars : SFFMParameters, optional
        Model configuration; defaults to :func:`sffm_get_default_model_parameters`.
    rng : numpy.random.Generator, optional
        For reproducibility. If given, overrides the phase field's own RNG.

    Returns
    -------
    2D ndarray of the same shape as ``tg_mat``, non-negative, whose sum equals
    ``tg_mat.sum()`` (mean slip preserved).
    """
    if sffm_pars is None:
        sffm_pars = sffm_get_default_model_parameters()
    tg = np.asarray(tg_mat, dtype=float)

    kx, ky = sffm_get_numerical_wavenumbers(tg)
    model_spec = sffm_pars.spectral_amplitude_fun(kx, ky, reg_par)

    # Phase-generating field -> phase from its FFT (Gallovic/Mai technique).
    if rng is not None:
        r_noise = rng.standard_normal(tg.size)
    else:
        r_noise = sffm_pars.random_phase_generating_field(tg.size)
    r_noise = np.asarray(r_noise, dtype=float).reshape(tg.shape)
    r_noise = r_noise * np.sign(r_noise.sum() if r_noise.sum() != 0 else 1.0)

    fake_phase = np.angle(np.fft.fft2(r_noise))
    tmp = model_spec * (np.cos(fake_phase) + 1j * np.sin(fake_phase))
    fake_data = np.real(np.fft.ifft2(tmp))

    fake = sffm_pars.negative_slip_removal_function(fake_data)

    if sffm_pars.recentre_slip:
        fake = sffm_recentre_slip(fake, tg)

    if sffm_pars.spatial_slip_decay in ("exponential", "gaussian"):
        max_r, max_c = np.unravel_index(np.argmax(fake), fake.shape)
        nr, nc = fake.shape
        xmat = np.broadcast_to(np.arange(nc), (nr, nc))
        ymat = np.broadcast_to(np.arange(nr)[:, None], (nr, nc))
        # reg_par[0] = kcx*dx, so cell-distance * reg_par gives distance/scale.
        dist2 = (((xmat - max_c) * reg_par[0]) ** 2
                 + ((ymat - max_r) * reg_par[1]) ** 2)
        if sffm_pars.spatial_slip_decay == "exponential":
            fake = fake * np.exp(-np.sqrt(dist2))
        else:
            fake = fake * np.exp(-dist2)
    elif sffm_pars.spatial_slip_decay != "none":
        raise ValueError("spatial_slip_decay not recognized")

    total = fake.sum()
    if total > 0:
        fake = fake / total * tg.sum()
    return fake


# ---------------------------------------------------------------------------
# Per-event random rupture length/width and corner wavenumbers
# ---------------------------------------------------------------------------

# rptha's own regression coefficients for log10(kcx), log10(kcy) against Mw
# (Davies et al. 2015), and the correlation between their residuals -- see
# sffm_make_random_lwkc_function's defaults in sffm_fit_simulate_earthquake.R.
_LOG10_KCX_REGRESSION = (-0.54, 2.03, 0.22)
_LOG10_KCY_REGRESSION = (-0.41, 1.18, 0.19)
_COR_KCX_KCY_RESIDUAL = 0.68


def sffm_make_random_lwkc_function(clip_random_parameter_ranges_to_2sd=False,
                                    relation="Strasser",
                                    force_deterministic=False):
    """Port of rptha's ``sffm_make_random_lwkc_function``.

    Returns a callable ``simulate_L_W_kcx_kcy(Mw, rng=None)`` -> a dict of
    arrays ``{"L", "W", "kcx", "kcy"}`` (one value per requested event),
    matching the R function's returned closure. ``Mw`` may be a scalar or
    array (one call generates ``len(Mw)`` events, as rptha's own vectorised
    interface does).

    Every event gets its OWN random rupture length/width (log-normal about
    the scaling relation's mean, using that relation's own log10 sigma --
    see :func:`pyptha_v12.scaling.Mw_2_rupture_size`'s ``detailed=True``
    output) and its own random PHYSICAL corner wavenumbers (log-normal
    regression against Mw, with correlated kcx/kcy residuals -- the
    correlation matters: a bigger-than-average kcx tends to come with a
    bigger-than-average kcy, not an independent draw). This is what makes
    every HS realisation at one magnitude look different from the others,
    beyond just the random phase field ``sffm_simulate`` itself draws.

    ``force_deterministic=True`` zeroes every sigma (random_scale=0 in the R
    code), so every event at a given Mw gets the exact scaling-relation mean
    -- useful for testing, matching R's own ``use_deterministic_LWkc`` path.
    """
    from .scaling import Mw_2_rupture_size

    random_scale = 0.0 if force_deterministic else 1.0

    def simulate_L_W_kcx_kcy(Mw, rng=None):
        rng = rng or np.random.default_rng()
        Mw_arr = np.atleast_1d(np.asarray(Mw, dtype=float))
        n = Mw_arr.size

        def rnorm_clipped(size):
            z = rng.standard_normal(size)
            if clip_random_parameter_ranges_to_2sd:
                z = np.clip(z, -2.0, 2.0)
            return z

        # log10 sigmas for width/length: Strasser's (and every other
        # relation's) sigma does not depend on Mw, so evaluating at any Mw
        # gives the same sigma -- rptha's own comment calls its Mw=7.5
        # "arbitrary" for exactly this reason.
        sizes = Mw_2_rupture_size(7.5, relation=relation, detailed=True)
        sigma_L = sizes.log10_sigmas["length"]
        sigma_W = sizes.log10_sigmas["width"]

        L_mw = np.array([Mw_2_rupture_size(float(m), relation=relation)["length"]
                         for m in Mw_arr])
        W_mw = np.array([Mw_2_rupture_size(float(m), relation=relation)["width"]
                         for m in Mw_arr])

        new_L = 10.0 ** (np.log10(L_mw) + sigma_L * rnorm_clipped(n) * random_scale)
        new_W = 10.0 ** (np.log10(W_mw) + sigma_W * rnorm_clipped(n) * random_scale)

        # Correlated kcx/kcy residuals (standard formula for two correlated
        # unit-variance normals from two independent ones).
        err_kcx = rng.standard_normal(n)
        err_kcy = (_COR_KCX_KCY_RESIDUAL * err_kcx
                   + math.sqrt(1.0 - _COR_KCX_KCY_RESIDUAL ** 2) * rng.standard_normal(n))
        if clip_random_parameter_ranges_to_2sd:
            err_kcx = np.clip(err_kcx, -2.0, 2.0)
            err_kcy = np.clip(err_kcy, -2.0, 2.0)

        a = _LOG10_KCX_REGRESSION
        b = _LOG10_KCY_REGRESSION
        kcx = 10.0 ** (a[0] * Mw_arr + a[1] + a[2] * err_kcx * random_scale)
        kcy = 10.0 ** (b[0] * Mw_arr + b[1] + b[2] * err_kcy * random_scale)

        return {"L": new_L, "W": new_W, "kcx": kcx, "kcy": kcy}

    return simulate_L_W_kcx_kcy


def rectangle_on_grid(n_rows, n_cols, num_rows, num_cols, centre_row, centre_col,
                       randomly_vary_around_target_centre=False, rng=None):
    """Port of rptha's ``rectangle_on_grid``, axis names spelled out.

    Places a ``num_rows`` x ``num_cols`` rectangle on an ``n_rows`` x
    ``n_cols`` grid (0-based here; R is 1-based), centred as close to
    ``(centre_row, centre_col)`` as the grid boundary allows. Returns
    ``(start_row, end_row, start_col, end_col)`` inclusive, 0-based.

    R's own version names its axes "L" (length, along-strike = columns
    here) and "W" (width, down-dip = rows here) and is called with those
    swapped per-argument (``rectangle_on_grid(dim(dx), c(num_W, num_L),
    c(peak_slip_row, peak_slip_col), ...)`` in
    sffm_make_events_on_discretized_source, then its result is unpacked as
    ``sW=bbox[1], eW=bbox[2], sL=bbox[3], eL=bbox[4]``) -- i.e. R's
    first (grid, num, centre) triple is really (down-dip / rows) and its
    second is (along-strike / columns). This port takes rows and columns as
    separate, explicitly-named parameters instead, so there is no axis
    ambiguity to preserve by convention; the boundary-handling logic below
    is otherwise identical to R's, one axis at a time.
    """
    rng = rng or np.random.default_rng()

    def _one_axis(n, num, centre):
        # 0-based version of R's 1-based logic (R's target_centre[i] - 1 ==
        # our centre; every +/-1 offset below cancels between the two
        # conventions except the final clamp bounds).
        if not randomly_vary_around_target_centre:
            if num % 2 == 0:
                s = centre - (num // 2 - rng.integers(0, 2))
            else:
                s = centre - num // 2
        else:
            s = centre - rng.integers(0, num)
        s = min(s, n - num)
        s = max(s, 0)
        e = min(s + num - 1, n - 1)
        return int(s), int(e)

    start_row, end_row = _one_axis(n_rows, num_rows, centre_row)
    start_col, end_col = _one_axis(n_cols, num_cols, centre_col)
    return start_row, end_row, start_col, end_col
