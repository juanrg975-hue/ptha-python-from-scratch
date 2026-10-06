"""Compare step 7b's synthetic HS and VAUS fields against PTHA18's OWN
published HS/VAUS catalogues -- a real, downloaded, official comparison,
not just a "no comparison possible" note.

Also used for VAUS
-------------------
The official VAUS netCDF (all_variable_uniform_slip_earthquake_events_
<zone>.nc) uses the EXACT SAME packed event_index_string/event_slip_string
schema as the HS one (confirmed against kermadectonga2's official pair:
same columns, same row count, VAUS rows are a deterministic function of the
matching HS row -- see step7b_stochastic_slip.py's module docstring). Every
function below (load_official_hs_events, official_events_to_grids,
compare_to_official) is schema-generic and works unchanged on either file;
only has_official_hs/has_official_vaus point at different filenames.

Why this comparison can exist at all
-------------------------------------
PTHA18 publishes not only its FAUS ("uniform slip") event table but also a
full heterogeneous-slip (HS, "stochastic slip") one:
``all_stochastic_slip_earthquake_events_<zone>.nc``, produced by the SAME
SFFM generator this package ports (rptha's sffm_fit_simulate_earthquake.R,
Davies et al. 2015's S_NCF method) via a wrapper this package does NOT port
(sffm_make_events_on_discretized_source -- it randomises rupture length,
width and corner wavenumber per event and generates ~15 HS realisations per
FAUS rupture on average, driven by an external script,
make_all_earthquake_events.R, not present in this repository).

Why this is a STATISTICAL comparison, not a field-by-field one
-----------------------------------------------------------------
SFFM is a random algorithm: even rptha's own R code, re-run with the same
corner wavenumbers on the same footprint, produces a DIFFERENT slip field
every call, because the phase spectrum is drawn from a fresh random-normal
field. The published netCDF does not record the random seed used per event
(sffm_events_to_table drops it), and R's and NumPy's random number
generators are not bit-compatible regardless. So no field this package
generates can, or should, be expected to numerically match any specific
published row.

What CAN be validated -- and what this module does -- is whether the
DISTRIBUTION of fields this port generates matches the distribution PTHA18's
own official fields show, for the same rupture size and corner wavenumber.
This is the same method rptha itself uses to validate the generator against
target data (sffm_slip_goodness_of_fit, comparing mean |FFT(slip)| over many
simulations against a target spectrum) -- applied here to compare TWO sets
of simulations (ours vs. the official ones) instead of one set against a
fitted target.

Data required (already present in this package for kermadectonga2 -- see
official_ptha_data/public_nc/)
-------------------------------------------------------------------------
  all_stochastic_slip_earthquake_events_<zone>.nc   the official HS table
  unit_source_statistics_<zone>.nc                  official unit-source
                                                      geometry, to place each
                                                      packed event back onto
                                                      a (downdip, alongstrike)
                                                      grid

Not fetched by default: these are large (the HS file alone is ~100-500 MB
depending on zone). download_official_hs_vaus() below fetches them from NCI
THREDDS (https://thredds.nci.org.au/thredds/fileServer/fj6/PTHA/AustPTHA_1/
SOURCE_ZONES/<zone>/TSUNAMI_EVENTS/all_stochastic_slip_earthquake_events_
<zone>.nc and .../all_variable_uniform_slip_earthquake_events_<zone>.nc),
the same path pattern and curl mechanics official_ptha_data/
fetch_official_inputs.py already uses for unit_source_statistics_<zone>.nc
-- but only when step7b_stochastic_slip.py is run with --download-official,
since a step_total.py run should not silently start a large download nobody
asked for. If the files are missing and --download-official was not passed,
the comparison is simply skipped -- see has_official_hs().
"""

from __future__ import annotations

import os
import subprocess

import numpy as np

THREDDS = "https://thredds.nci.org.au/thredds/fileServer/fj6/PTHA/AustPTHA_1"


def _decode_packed(var, i):
    """Decode one row of a netCDF4 fixed-width byte-array string variable."""
    raw = var[i]
    text = raw.tobytes().decode("utf-8", "ignore")
    return text.split("\x00")[0]


def has_official_hs(zone, official_nc_dir):
    """Path to the official stochastic-slip netCDF for `zone`, or None."""
    path = os.path.join(
        official_nc_dir, f"all_stochastic_slip_earthquake_events_{zone}.nc")
    return path if os.path.exists(path) else None


def has_official_vaus(zone, official_nc_dir):
    """Path to the official variable-uniform-slip (VAUS) netCDF for `zone`,
    or None. Same packed event_index_string/event_slip_string schema as the
    HS file (confirmed against kermadectonga2's official netCDF pair), so
    every function below works unchanged on either file."""
    path = os.path.join(
        official_nc_dir, f"all_variable_uniform_slip_earthquake_events_{zone}.nc")
    return path if os.path.exists(path) else None


def download_official_hs_vaus(zone, official_nc_dir):
    """Fetch the official HS and VAUS netCDFs for `zone` from NCI THREDDS
    into `official_nc_dir`, unless already present. Same download mechanics
    as fetch_official_inputs.py's download(): write to a .part file first so
    an interrupted download never leaves a truncated file a later run would
    silently trust as complete.

    A 404 (curl exit 22) means PTHA18 never published that file for this
    zone -- not every zone has HS/VAUS catalogues published, even when its
    FAUS one exists. That is reported, not raised, so the caller can still
    proceed with whichever of the two files did fetch successfully.
    """
    os.makedirs(official_nc_dir, exist_ok=True)
    for fname in (f"all_stochastic_slip_earthquake_events_{zone}.nc",
                  f"all_variable_uniform_slip_earthquake_events_{zone}.nc"):
        dest = os.path.join(official_nc_dir, fname)
        if os.path.exists(dest):
            mb = os.path.getsize(dest) / 1e6
            print(f"      have {fname} ({mb:,.1f} MB) - skipping")
            continue
        url = f"{THREDDS}/SOURCE_ZONES/{zone}/TSUNAMI_EVENTS/{fname}"
        tmp = dest + ".part"
        print(f"      downloading {fname}\n         {url}")
        try:
            subprocess.run(["curl", "-fL", "--progress-bar", "-o", tmp, url],
                            check=True)
        except FileNotFoundError:
            print("      curl not found; cannot download official HS/VAUS "
                  "catalogues -- skipping")
            return
        except subprocess.CalledProcessError as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            if e.returncode == 22:
                print(f"      {fname} not published for '{zone}' (404) -- "
                      f"skipping")
            else:
                print(f"      download failed for {fname} (curl exit "
                      f"{e.returncode}) -- skipping")
            continue
        os.replace(tmp, dest)
        print(f"      saved {dest} ({os.path.getsize(dest) / 1e6:,.1f} MB)")


def load_official_hs_events(nc_path, target_mw, mw_tol=0.05, max_events=None,
                             rng=None):
    """Load official HS rows near `target_mw`.

    `max_events=None` (the default) loads EVERY official row within
    `mw_tol` of `target_mw` -- for a well-modelled zone like kermadectonga2
    this can be several thousand rows at the small-magnitude end, which is
    the point: a statistical comparison is only as good as the official
    sample size backing it. Pass an integer to subsample instead (e.g. for
    a quick interactive check).

    Returns a list of dicts: {mw, downdip_idx, alongstrike_idx, slip}
    (0-based grid indices, matched against the geometry table's
    downdip_number/alongstrike_number - 1) plus the two corner-wavenumber
    columns, so a caller can regenerate comparable synthetic fields with the
    SAME reg_par this official sample actually used.
    """
    import netCDF4

    with netCDF4.Dataset(nc_path) as ds:
        mw = np.asarray(ds.variables["Mw"][:], dtype=float)
        close = np.where(np.abs(mw - target_mw) <= mw_tol)[0]
        if close.size == 0:
            return []
        if max_events is not None and close.size > max_events:
            rng = rng or np.random.default_rng(0)
            close = rng.choice(close, size=max_events, replace=False)

        idx_var = ds.variables["event_index_string"]
        slip_var = ds.variables["event_slip_string"]
        kcx = np.asarray(ds.variables["physical_corner_wavenumber_x"][:], dtype=float)
        kcy = np.asarray(ds.variables["physical_corner_wavenumber_y"][:], dtype=float)

        out = []
        for i in close:
            i = int(i)
            idx_str = _decode_packed(idx_var, i)
            slip_str = _decode_packed(slip_var, i)
            ids = [int(p) for p in idx_str.split("-") if p]
            slips = [float(p) for p in slip_str.split("_") if p]
            if len(ids) != len(slips) or not ids:
                continue
            out.append({
                "mw": float(mw[i]), "unit_source_ids_1based": ids,
                "slip": np.asarray(slips, dtype=float),
                "kcx": float(kcx[i]), "kcy": float(kcy[i]),
            })
        return out


def official_events_to_grids(events, downdip_number, alongstrike_number):
    """Place each official event's packed (id, slip) pairs onto a dense
    (downdip x alongstrike) matrix, using the SAME unit-source-id ->
    (downdip, alongstrike) mapping the official geometry table defines.

    `downdip_number`/`alongstrike_number` are 1-based arrays indexed the
    same way as the official event_index_string (subfault_number 1..N, in
    file row order) -- i.e. read straight from
    unit_source_statistics_<zone>.nc, not from this run's own from-scratch
    mesh (the whole point is comparing against the OFFICIAL mesh's events).

    Returns a list of 2D arrays, one per event, each cropped to its own
    rupture's bounding box (like step 7b's own templates) so grid size does
    not vary the spectrum's sampled wavenumbers between events of different
    footprints.
    """
    dd = np.asarray(downdip_number, dtype=int)
    ask = np.asarray(alongstrike_number, dtype=int)
    grids = []
    for ev in events:
        ids0 = np.asarray(ev["unit_source_ids_1based"], dtype=int) - 1
        if ids0.max(initial=-1) >= dd.size or ids0.min(initial=0) < 0:
            continue
        dd_ev = dd[ids0]
        ask_ev = ask[ids0]
        dd0, as0 = dd_ev.min(), ask_ev.min()
        nd, na = dd_ev.max() - dd0 + 1, ask_ev.max() - as0 + 1
        g = np.zeros((nd, na))
        g[dd_ev - dd0, ask_ev - as0] = ev["slip"]
        grids.append(g)
    return grids


def spectral_signature(grid):
    """A single event's amplitude spectrum, radially summarised.

    Matches the quantity rptha's own sffm_slip_goodness_of_fit compares
    (mean |FFT(slip)|), reduced further here to a 1D radial profile so
    fields of different shapes can still be compared via their spectral
    slope/shape rather than requiring identical grid dimensions.
    """
    amp = np.abs(np.fft.fft2(grid))
    ny, nx = grid.shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    k = np.sqrt(ky ** 2 + kx ** 2)
    nbins = max(min(ny, nx) // 2, 2)
    edges = np.linspace(0, k.max() + 1e-12, nbins + 1)
    profile = np.zeros(nbins)
    for b in range(nbins):
        mask = (k >= edges[b]) & (k < edges[b + 1])
        if mask.any():
            profile[b] = amp[mask].mean()
    total = profile.sum()
    return profile / total if total > 0 else profile


def summary_stats(grid):
    """Scalar shape statistics for one slip field: peak, active-cell
    fraction, and a spatial-concentration index (fraction of total slip
    held by the single largest cell) -- cheap, interpretable numbers a
    report reader can compare without needing to read a spectrum."""
    nz = grid[grid > 0]
    total = grid.sum()
    return {
        "peak_slip_m": float(grid.max()) if grid.size else 0.0,
        "active_cells": int((grid > 0).sum()),
        "total_cells": int(grid.size),
        "concentration": float(grid.max() / total) if total > 0 else 0.0,
    }


COMMON_SPECTRUM_BINS = 16


def field_features(grids):
    """Everything compare_to_official needs from a set of fields, one row
    per field: [peak_slip_m, active_cells, concentration, 16 spectrum bins].

    Step 7b saves these for its own fields (outputs/*_slip_fields/
    comparison_features.npz), so step 8 can compare them with PTHA18's
    catalogue without step 7b reading anything of PTHA18.
    """
    rows = []
    xq = np.linspace(0, 1, COMMON_SPECTRUM_BINS)
    for g in grids:
        s = summary_stats(g)
        prof = spectral_signature(g)
        rows.append(np.concatenate((
            [s["peak_slip_m"], s["active_cells"], s["concentration"]],
            np.interp(xq, np.linspace(0, 1, prof.size), prof))))
    return np.asarray(rows, dtype=float).reshape(len(rows), 3 + COMMON_SPECTRUM_BINS)


def aggregate_features(features):
    """The per-side means compare_to_official reports, from field_features()."""
    f = np.asarray(features, dtype=float)
    n = int(f.shape[0])
    return {
        "n": n,
        "mean_peak_slip_m": float(np.mean(f[:, 0])) if n else float("nan"),
        "mean_active_cells": float(np.mean(f[:, 1])) if n else float("nan"),
        "mean_concentration": float(np.mean(f[:, 2])) if n else float("nan"),
        "mean_spectrum": np.mean(f[:, 3:], axis=0) if n else np.zeros(COMMON_SPECTRUM_BINS),
    }


def compare_features(synthetic_features, official_features):
    """compare_to_official, from field_features() of each side."""
    syn = aggregate_features(synthetic_features)
    off = aggregate_features(official_features)
    spectral_distance = float(np.mean(np.abs(syn["mean_spectrum"] - off["mean_spectrum"]))) \
        if syn["n"] and off["n"] else float("nan")
    return {"synthetic": syn, "official": off, "spectral_distance": spectral_distance}


def compare_to_official(synthetic_grids, official_grids):
    """Compare a set of synthetic HS fields to a set of official ones for
    the SAME magnitude (and ideally similar footprint size).

    Returns a dict of aggregate statistics for each side plus a simple
    spectral-distance number (mean absolute difference between the two
    sides' average radial spectra, both resampled onto a common 16-bin
    axis so differing grid shapes do not block the comparison).
    """
    return compare_features(field_features(synthetic_grids),
                            field_features(official_grids))
