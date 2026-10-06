"""Uniform-slip earthquake rupture generation.

Python port of ``rptha/R/rupture_events.R``:

    get_all_earthquake_events_of_magnitude_Mw   - all fixed-size uniform-slip
        ruptures of one magnitude on a discretised source
    get_all_earthquake_events                   - the same swept over a range
        of magnitudes, returned as one event table
    get_unit_source_indices_in_event            - parse an event_index_string
    coverage_weights                            - v10_q: the q weight for
        ruptures of rupture_size="local"

A rupture ("event") is a rectangular block of adjacent unit sources whose
number-of-cells along strike and down dip is chosen so the total area matches
the Strasser (or other) scaling relation for that magnitude, with the aspect
ratio picked to best match the relation. Every valid placement of that block
on the grid is enumerated. Slip is uniform and set so the seismic moment
equals M0(Mw) exactly.

v10: ``rupture_size="local"`` (opt-in; the default ``"rptha"`` is rptha's
rule above, unchanged) sizes the block at EACH placement from the real km of
the cells there, instead of one cell count per magnitude from the zone's
mean cell size. rptha's rule assumes cells of roughly equal size (PTHA18's
are ~50 x 50 km); on a mesh whose cells differ a lot (calabria2: 377 to
4042 km2), one cell count gives ruptures of very different areas for the
same Mw. See :func:`_local_blocks`.

v10_q: with ``"local"``, the engine also multiplies PTHA18's conditional
probability weight by :func:`coverage_weights`, because ``"local"`` makes one
rupture per starting cell, so where cells are small there are more ruptures
per km2 (rptha's area weight used to compensate that by accident).

Works directly from a unit-source summary-statistics table (the dict produced
by :func:`pyptha.unit_sources.discretized_source_approximate_summary_statistics`).
"""

from __future__ import annotations

import itertools

import numpy as np

from .scaling import M0_2_Mw, Mw_2_rupture_size, slip_from_Mw_area_mu


def _local_blocks(desired, length, width, downdip, alongstrike):
    """v10 ``rupture_size="local"``: one block size per placement.

    For each top-left cell, and each number of rows ``nw``, the number of
    columns ``nl`` whose summed cell area is closest to the scaling
    relation's area (log scale) is found; of those, the ``(nl, nw)`` with the
    smallest ``|log10(area error)| + |log10(aspect error)|`` is kept. Lengths
    and widths are the block's own km, as rptha reports them per event
    (summed cell length / rows, summed cell width / columns).

    A placement whose best block would need cells beyond the far edge of the
    mesh is skipped, as rptha skips placements where its block does not fit:
    the search may use one virtual column (row) beyond the edge, a copy of
    the last one, and a placement that picks it is dropped. The first column
    (row) has no virtual neighbour, so a rupture longer (wider) than the
    whole zone becomes the full zone length (width) at the first placement
    only, with the other dimension stretched to recover the area -- rptha's
    own fallback (``nlength > nstrike`` / ``nwidth > ndip``). For the
    full-length case the rows are then chosen by area alone, as rptha does;
    with the aspect term, a full-length block starting lower down could keep
    fewer, smaller rows and come out far below the target area.

    Skipping alone would leave some cells of the mesh, mostly next to the
    far edges, out of every rupture of some magnitudes (rptha's blocks cover
    every cell: its last placement always ends flush with the edge). So each
    cell no block covers then gets one more block: of all the blocks that
    contain that cell and fit on the mesh, the one with the smallest cost
    above (the whole zone always qualifies).

    Returns a list of ``(dipIndex, strikeIndex, nlength, nwidth)``, 1-based,
    sorted in rptha's placement order (down-dip index fastest).
    """
    ndip, nstrike = int(downdip.max()), int(alongstrike.max())
    L = np.zeros((ndip, nstrike))
    W = np.zeros((ndip, nstrike))
    L[downdip - 1, alongstrike - 1] = length
    W[downdip - 1, alongstrike - 1] = width
    logA, log_aspect = (np.log10(desired["area"]),
                        np.log10(desired["length"] / desired["width"]))

    def prefix(a):
        p = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
        p[1:, 1:] = a.cumsum(0).cumsum(1)
        return p

    # One virtual row and column beyond the far edges (copies of the last).
    Lp = np.pad(L, ((0, 1), (0, 1)), mode="edge")
    Wp = np.pad(W, ((0, 1), (0, 1)), mode="edge")
    PA, PL, PW = prefix(Lp * Wp), prefix(Lp), prefix(Wp)

    def block_sum(P, d0, s0, nl, nw):
        """sum over the block(s) with 0-based top-left (d0, s0), nl x nw"""
        return (P[d0 + nw, s0 + nl] - P[d0, s0 + nl]
                - P[d0 + nw, s0] + P[d0, s0])

    def cost_of(d0, s0, nl, nw):
        """area and aspect cost of blocks (0-based top-left, sizes arrays)"""
        aspect = (block_sum(PL, d0, s0, nl, nw) / nw) / (block_sum(PW, d0, s0, nl, nw) / nl)
        return (np.abs(np.log10(block_sum(PA, d0, s0, nl, nw)) - logA)
                + np.abs(np.log10(aspect) - log_aspect))

    def grow(anchors):
        """Best block from each 0-based (d0, s0), growing right and down."""
        out = []
        for d0, s0 in anchors:
            # rows/columns allowed, plus the virtual one unless at the start
            max_nl = nstrike - s0 + (1 if s0 > 0 else 0)
            max_nw = ndip - d0 + (1 if d0 > 0 else 0)
            nl = np.arange(1, max_nl + 1)
            cands = []  # (cost, nl, nw, area error, block length km)
            for nw in range(1, max_nw + 1):
                area_err = np.abs(np.log10(block_sum(PA, d0, s0, nl, nw)) - logA)
                k = int(np.argmin(area_err))
                block_length = block_sum(PL, d0, s0, nl[k], nw) / nw
                aspect = block_length / (block_sum(PW, d0, s0, nl[k], nw) / nl[k])
                cands.append((area_err[k] + abs(np.log10(aspect) - log_aspect),
                              int(nl[k]), nw, area_err[k], block_length))
            best = min(cands, key=lambda c: c[0])
            if (s0 == 0 and best[1] == nstrike
                    and best[4] < desired["length"]):
                # Longer than the whole zone: full length, rows by area alone
                # (rptha's nlength > nstrike branch).
                best = min((c for c in cands if c[1] == nstrike),
                           key=lambda c: c[3])
            _, bl, bw, _, _ = best
            if s0 + bl > nstrike or d0 + bw > ndip:
                continue  # needs the virtual column/row: does not fit here
            out.append((d0, s0, bl, bw))
        return out

    blocks = grow([(d, s) for s in range(nstrike) for d in range(ndip)])
    covered = np.zeros((ndip, nstrike), bool)
    for d0, s0, bl, bw in blocks:
        covered[d0:d0 + bw, s0:s0 + bl] = True
    for s in range(nstrike):
        for d in range(ndip):
            if covered[d, s]:
                continue
            # every block that contains (d, s) and fits on the mesh
            d0, s0, nl, nw = (a.ravel() for a in np.meshgrid(
                np.arange(d + 1), np.arange(s + 1),
                np.arange(1, nstrike + 1), np.arange(1, ndip + 1),
                indexing="ij"))
            ok = ((s0 + nl > s) & (d0 + nw > d)
                  & (s0 + nl <= nstrike) & (d0 + nw <= ndip))
            d0, s0, nl, nw = d0[ok], s0[ok], nl[ok], nw[ok]
            k = int(np.argmin(cost_of(d0, s0, nl, nw)))
            blocks.append((int(d0[k]), int(s0[k]), int(nl[k]), int(nw[k])))
            covered[d0[k]:d0[k] + nw[k], s0[k]:s0[k] + nl[k]] = True
    blocks = sorted(set(blocks), key=lambda b: (b[1], b[0], b[2], b[3]))
    return [(d + 1, s + 1, bl, bw) for d, s, bl, bw in blocks]


def get_all_earthquake_events_of_magnitude_Mw(Mw, unit_source_stats,
                                              mu=3.0e10, constant=9.05,
                                              relation="Strasser",
                                              rupture_size="rptha"):
    """All fixed-size uniform-slip ruptures of magnitude ``Mw``.

    ``unit_source_stats`` is the summary-statistics dict (arrays keyed by
    lon_c, lat_c, depth, max_depth, length, width, downdip_number,
    alongstrike_number, subfault_number). Returns a dict describing the
    ruptures, including per-event area/slip/depth statistics and the list of
    unit-source indices (0-based) in each event.

    ``rupture_size``: "rptha" (default, rptha's rule: one block size for the
    magnitude) or "local" (v10, a block size per placement, see
    :func:`_local_blocks`). With "local", ``event_dim`` is None and
    ``event_dims`` holds each event's own ``{"length", "width"}`` in cells.
    """
    if rupture_size not in ("rptha", "local"):
        raise ValueError(f"rupture_size must be 'rptha' or 'local', "
                         f"not {rupture_size!r}")
    subn = np.asarray(unit_source_stats["subfault_number"], dtype=int)
    if not np.array_equal(subn, np.arange(1, subn.size + 1)):
        raise ValueError("subfault_number must run 1..N in order")

    length = np.asarray(unit_source_stats["length"], dtype=float)
    width = np.asarray(unit_source_stats["width"], dtype=float)
    depth = np.asarray(unit_source_stats["depth"], dtype=float)
    max_depth = np.asarray(unit_source_stats["max_depth"], dtype=float)
    downdip = np.asarray(unit_source_stats["downdip_number"], dtype=int)
    alongstrike = np.asarray(unit_source_stats["alongstrike_number"], dtype=int)

    rupture_stats = Mw_2_rupture_size(Mw, relation=relation, detailed=True, CI_sd=2.0)
    desired = rupture_stats.values  # dict area/width/length (km, km^2)
    M0 = M0_2_Mw(Mw, inverse=True, constant=constant)

    mean_area = float(np.mean(length * width))
    mean_width = float(np.mean(width))
    mean_length = float(np.mean(length))

    ndip = int(downdip.max())
    nstrike = int(alongstrike.max())
    desired_subfault_count = max(round(desired["area"] / mean_area), 1)

    nlength = int(np.ceil(desired["length"] / mean_length))
    nwidth = int(np.ceil(desired["width"] / mean_width))

    if nlength > nstrike:
        nlength = nstrike
        nwidth = min(max(round(desired_subfault_count / nlength), 1), ndip)
    elif nwidth > ndip:
        nwidth = ndip
        nlength = min(max(round(desired_subfault_count / nwidth), 1), nstrike)
    else:
        # Try four length/width candidates, pick the best aspect ratio (in
        # log space, following Strasser), exactly as the R code does.
        desired_aspect = desired["length"] / desired["width"]
        l0 = [0, 0, 0, 0]
        w0 = [0, 0, 0, 0]
        l0[0] = min(max(int(np.floor(desired["length"] / mean_length)), 1), nstrike)
        l0[1] = min(max(int(np.ceil(desired["length"] / mean_length)), 1), nstrike)
        w0[0] = min(max(round(desired_subfault_count / l0[0]), 1), ndip)
        w0[1] = min(max(round(desired_subfault_count / l0[1]), 1), ndip)
        w0[2] = min(max(int(np.floor(desired["width"] / mean_width)), 1), ndip)
        w0[3] = min(max(int(np.ceil(desired["width"] / mean_width)), 1), ndip)
        l0[2] = min(max(round(desired_subfault_count / w0[2]), 1), nstrike)
        l0[3] = min(max(round(desired_subfault_count / w0[3]), 1), nstrike)

        aspect_err = np.abs(
            np.log10((np.array(l0) * mean_length) / (np.array(w0) * mean_width))
            - np.log10(desired_aspect))
        k = int(np.argmin(aspect_err))
        nwidth, nlength = w0[k], l0[k]

    nlength = min(nlength, nstrike)
    nwidth = min(nwidth, ndip)

    # Enumerate every top-left placement of the (nlength x nwidth) block, in
    # rptha's order: expand.grid(dipIndex, strikeIndex) varies the down-dip
    # index fastest (rupture_events.R:128). v8: earlier versions varied the
    # along-strike index fastest, which gave the same events in a different
    # row order, so event numbers did not match PTHA18's event tables.
    dip_range = range(1, ndip - nwidth + 2)
    strike_range = range(1, nstrike - nlength + 2)
    if rupture_size == "local":
        blocks = _local_blocks(desired, length, width, downdip, alongstrike)
    else:
        blocks = [(d, s, nlength, nwidth)
                  for s, d in itertools.product(strike_range, dip_range)]
    topleft = [(d, s) for d, s, _, _ in blocks]  # (dipIndex, strikeIndex)

    n_events = len(topleft)
    ev = {
        "area": np.zeros(n_events), "mean_length": np.zeros(n_events),
        "mean_width": np.zeros(n_events), "slip": np.zeros(n_events),
        "Mw": np.full(n_events, float(Mw)), "mean_depth": np.zeros(n_events),
        "max_depth": np.zeros(n_events),
    }
    event_indices = []

    for i, (dip_i, strike_i, nl, nw) in enumerate(blocks):
        want_dd = set(dip_i + np.arange(nw))
        want_as = set(strike_i + np.arange(nl))
        idx = np.where(np.isin(downdip, list(want_dd))
                       & np.isin(alongstrike, list(want_as)))[0]
        if idx.size != nl * nw:
            raise ValueError(f"Mw {Mw}: incorrect number of subfaults "
                             f"({idx.size} != {nl * nw})")
        event_indices.append(idx)

        ll = length[idx]
        ww = width[idx]
        ev["area"][i] = np.sum(ll * ww)
        ev["mean_length"][i] = np.sum(ll) / nw
        ev["mean_width"][i] = np.sum(ww) / nl
        ev["slip"][i] = slip_from_Mw_area_mu(Mw, ev["area"][i], mu, constant=constant)
        ev["mean_depth"][i] = np.mean(depth[idx])
        ev["max_depth"][i] = np.max(max_depth[idx])

        local_M0 = ev["slip"][i] * (ev["area"][i] * 1e6) * mu
        if not np.isclose(local_M0, M0):
            raise ValueError("seismic moment mismatch for a generated event")

    local = rupture_size == "local"
    return {
        "topleft_indices": topleft,
        "event_dim": None if local else {"length": nlength, "width": nwidth},
        "event_dims": [{"length": nl, "width": nw} for _, _, nl, nw in blocks],
        "desired_subfault_count": desired_subfault_count,
        "actual_subfault_count": (np.array([nl * nw for _, _, nl, nw in blocks])
                                  if local else nlength * nwidth),
        "desired_ALW": desired,
        "event_statistics": ev,
        "event_indices": event_indices,
        "Mw": float(Mw),
    }


def get_all_earthquake_events(unit_source_statistics, Mmin=7.5, Mmax=9.6,
                              dMw=0.1, mu=3.0e10, constant=9.05,
                              source_zone_name=None, relation="Strasser",
                              rupture_size="rptha"):
    """All uniform-slip ruptures over a magnitude range, as one event table.

    ``rupture_size`` is passed to :func:`get_all_earthquake_events_of_magnitude_Mw`.

    Returns a dict of equal-length arrays (the "big event table"), with:
    ``Mw, area, slip, mean_length, mean_width, mean_depth, max_depth,
    event_index_string, sourcename`` and a Python list ``event_indices``
    (0-based unit-source indices per event). ``event_index_string`` uses
    1-based indices joined by '-' with a trailing '-', matching rptha.
    """
    mws = np.round(np.arange(Mmin, Mmax + 1e-9, dMw), 6)

    cols = {k: [] for k in ("Mw", "area", "slip", "mean_length",
                            "mean_width", "mean_depth", "max_depth")}
    event_index_string = []
    event_indices = []

    for mw in mws:
        e = get_all_earthquake_events_of_magnitude_Mw(
            mw, unit_source_statistics, mu=mu, constant=constant, relation=relation,
            rupture_size=rupture_size)
        es = e["event_statistics"]
        n = es["Mw"].size
        for k in cols:
            cols[k].append(es[k])
        for idx in e["event_indices"]:
            # 1-based, '-' separated with trailing '-', as in rptha.
            event_index_string.append("".join(f"{j + 1}-" for j in idx))
            event_indices.append(idx)

    table = {k: np.concatenate(v) for k, v in cols.items()}
    table["event_index_string"] = np.array(event_index_string, dtype=object)
    table["sourcename"] = np.array([source_zone_name] * len(event_index_string),
                                   dtype=object)
    table["event_indices"] = event_indices
    return table


def get_unit_source_indices_in_event(event_index_string):
    """Parse an ``event_index_string`` into 0-based unit-source indices.

    Inverse of the string encoding in :func:`get_all_earthquake_events`.
    Accepts the string itself (e.g. ``"3-4-5-"``) and returns a numpy array of
    0-based integer indices into the unit-source table.
    """
    parts = [p for p in str(event_index_string).split("-") if p != ""]
    return np.array([int(p) - 1 for p in parts], dtype=int)


def coverage_weights(event_Mw, event_indices, n_unit_sources):
    """v10_q: the q weight of each rupture, for ``rupture_size="local"``.

    ``q_e`` = mean, over the cells of rupture ``e``, of 1 / (number of
    ruptures of the same magnitude that contain that cell). Multiplying a
    magnitude's conditional probabilities by it removes the effect of how
    many ruptures happen to overlap a place: a part of the zone covered by
    three times as many ruptures (smaller cells, one rupture per starting
    cell) gives each of them a third of the weight, so both parts get the
    same rate for the same area and convergence. With rptha's rule on equal
    cells every interior cell is covered by the same number of ruptures, so
    q is the same for all interior ruptures.

    ``event_Mw`` (one per event, already rounded as in the event table),
    ``event_indices`` (0-based unit-source indices per event, as
    :func:`get_all_earthquake_events` returns them) and ``n_unit_sources``.
    Returns a numpy array, one q per event.
    """
    event_Mw = np.asarray(event_Mw, dtype=float)
    q = np.empty(event_Mw.size)
    for mw in np.unique(event_Mw):
        k = np.where(event_Mw == mw)[0]
        count = np.zeros(n_unit_sources)
        for j in k:
            count[event_indices[j]] += 1
        for j in k:
            q[j] = np.mean(1.0 / count[event_indices[j]])
    return q
