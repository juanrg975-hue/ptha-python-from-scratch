"""v12: read an external quadrilateral mesh into the pipeline's node array.

The input is a text file with one line per subfault (unit source) and 12
numbers per line: the four corners of the cell, each as lon, lat, depth
(``lon1 lat1 dep1 lon2 lat2 dep2 lon3 lat3 dep3 lon4 lat4 dep4``), for
example ``alaskaaleutians_quadrilateral_coors.dat``. The four corners must go
round the cell (each corner next to the following one); where the ring
starts and its direction do not matter, nor does the order of the lines.

What the pipeline needs is a STRUCTURED mesh, rows down dip by columns along
strike, the same node array the SLAB2 steps build:
``grid[i, :, j] = (lon, lat, depth_km)``, row 0 at the trench, depth positive
down, columns ordered so the fault dips to the right of the strike (rptha's
convention). :func:`read_quadrilateral_mesh` finds that structure from the
corners the cells share; nothing else is assumed:

1. depth units and sign: metres if any |depth| > 200, km otherwise (a
   subduction interface never reaches 200 km); negative-down values are
   turned positive-down. Both can be forced with ``depth_units``.
2. every cell's top, bottom and side edges, from its connections: the
   orientation is carried from cell to cell across shared edges (the edge a
   cell shares with the one below it is its bottom and that cell's top), so
   all cells agree; depth decides only once, for the whole mesh, which pair
   of opposite edges goes down dip (largest depth change summed over the
   cells) and which side is up. (Until 2026-10-06 each cell's shallowest
   edge was its top, which failed on fine meshes of flat or skewed slabs,
   where a few cells deepen more across strike than down dip.)
3. two cells are down-dip neighbours when one's bottom edge is the other's
   top edge, along-strike neighbours when they share a side edge (corners
   matched to ``tol_deg`` and ``tol_km``).
4. the top row is the cells with nothing above them; they must form one
   chain of side neighbours, and every column a chain of the same length.
5. the columns are reversed if needed so the fault dips to the right of the
   strike.

A mesh that is not a complete rows x columns lattice (triangles, a hole, a
column shorter than the others, cells that do not share their corners)
raises ValueError saying what is wrong.
"""

from __future__ import annotations

import numpy as np


def _key(lon, lat, dep, tol_deg, tol_km):
    return (round(lon / tol_deg), round(lat / tol_deg), round(dep / tol_km))


_RING = [(0, 1), (1, 2), (2, 3), (3, 0)]
_CYCLE = ["tl", "tr", "br", "bl"]  # going round a cell
# a corner's label in the cell across a shared edge: a top/bottom edge keeps
# left/right and swaps top/bottom, a side edge the other way round
_ACROSS_TB = {"tl": "bl", "tr": "br", "bl": "tl", "br": "tr"}
_ACROSS_LR = {"tl": "tr", "tr": "tl", "bl": "br", "br": "bl"}
_EDGES = {"top": ("tl", "tr"), "bottom": ("bl", "br"), "left": ("tl", "bl"), "right": ("tr", "br")}


def _orient_cells(c, k, edge):
    """Top, bottom and side edges of every cell, from the cells' connections.

    Each cell's corners are labelled tl, tr, br, bl so that neighbours agree:
    the edge a cell shares with the one below it is its bottom edge and that
    cell's top edge, and so on. The labels are carried from cell to cell
    across shared edges, so the whole mesh gets one orientation; only then
    is depth used, once for the whole mesh: the pair of opposite edges whose
    depth differs most, summed over all cells, is top/bottom, the shallower
    side the top. A per-cell rule (each cell's shallowest edge on top) fails
    on fine meshes of flat or skewed slabs, where a few cells deepen more
    across strike than down dip.
    """
    n = c.shape[0]
    by_edge = {}
    for i in range(n):
        for p, q in _RING:
            by_edge.setdefault(edge(i, (p, q)), []).append((i, p, q))
    if any(len(v) > 2 for v in by_edge.values()):
        raise ValueError("an edge is shared by more than two cells: not a rows x columns mesh")

    labels = [None] * n  # labels[i][p] = label of corner p of cell i

    def fill(i, a, la, b, lb):
        """Labels of cell i from two adjacent corners a, b labelled la, lb."""
        step = 1 if (b - a) % 4 == 1 else -1
        cyc = 1 if (_CYCLE.index(lb) - _CYCLE.index(la)) % 4 == 1 else -1
        out = [None] * 4
        for m in range(4):
            out[(a + step * m) % 4] = _CYCLE[(_CYCLE.index(la) + cyc * m) % 4]
        return out

    for seed in range(n):
        if labels[seed] is not None:
            continue
        if seed > 0:
            raise ValueError("the cells do not form one connected mesh")
        labels[0] = list(_CYCLE)
        queue = [0]
        while queue:
            i = queue.pop()
            for p, q in _RING:
                others = [e for e in by_edge[edge(i, (p, q))] if e[0] != i]
                if not others:
                    continue
                j = others[0][0]
                lp, lq = labels[i][p], labels[i][q]
                across = _ACROSS_TB if {lp, lq} in ({"tl", "tr"}, {"bl", "br"}) else _ACROSS_LR
                kp, kq = k(i, p), k(i, q)
                pj = [m for m in range(4) if k(j, m) == kp]
                qj = [m for m in range(4) if k(j, m) == kq]
                if len(pj) != 1 or len(qj) != 1:
                    raise ValueError(f"cells {i + 1} and {j + 1} do not share their corners")
                new = fill(j, pj[0], across[lp], qj[0], across[lq])
                if labels[j] is None:
                    labels[j] = new
                    queue.append(j)
                elif labels[j] != new:
                    raise ValueError(f"cell {j + 1}: its neighbours disagree on its "
                                     f"orientation: not a rows x columns mesh")

    # the whole mesh: which pair of opposite edges goes down dip, and which way
    def edge_depth(i, name):
        a, b = _EDGES[name]
        return c[i, [labels[i].index(a), labels[i].index(b)], 2].mean()

    d_tb = sum(edge_depth(i, "bottom") - edge_depth(i, "top") for i in range(n))
    d_lr = sum(edge_depth(i, "right") - edge_depth(i, "left") for i in range(n))
    if max(abs(d_tb), abs(d_lr)) <= 0:
        raise ValueError("the mesh is flat: cannot tell down dip from along strike")
    if abs(d_lr) > abs(d_tb):  # turn a quarter: left becomes top, right bottom
        rot = {"bl": "tl", "tl": "tr", "tr": "br", "br": "bl"}
        labels = [[rot[x] for x in lab] for lab in labels]
        d_tb = d_lr
    if d_tb < 0:  # upside down
        labels = [[_ACROSS_TB[x] for x in lab] for lab in labels]

    top, bot, sides = [], [], []
    for lab in labels:
        ring_edge = {frozenset((lab[p], lab[q])): (p, q) for p, q in _RING}
        top.append(ring_edge[frozenset(_EDGES["top"])])
        bot.append(ring_edge[frozenset(_EDGES["bottom"])])
        sides.append([ring_edge[frozenset(_EDGES["right"])], ring_edge[frozenset(_EDGES["left"])]])
    return top, bot, sides


def read_quadrilateral_mesh(path, depth_units="auto", tol_deg=1e-6, tol_km=1e-3,
                            log=print):
    """Node array (rows+1, 3, columns+1) of lon, lat, depth_km from ``path``."""
    a = np.loadtxt(path, ndmin=2)
    if a.shape[1] != 12:
        raise ValueError(f"{path}: expected 12 numbers per line (4 corners x "
                         f"lon, lat, depth), found {a.shape[1]}")
    c = a.reshape(-1, 4, 3).copy()
    n = c.shape[0]

    dep = c[..., 2]
    if depth_units == "auto":
        depth_units = "m" if np.nanmax(np.abs(dep)) > 200 else "km"
    if depth_units not in ("m", "km"):
        raise ValueError(f"depth_units must be 'auto', 'm' or 'km', not {depth_units!r}")
    dep = dep / (1000.0 if depth_units == "m" else 1.0)
    if np.nanmedian(dep) < 0:
        dep = -dep
    c[..., 2] = dep
    if np.any(dep < -1e-6):
        raise ValueError("depths of both signs: cannot tell down from up")

    k = lambda i, p: _key(*c[i, p], tol_deg, tol_km)  # noqa: E731
    edge = lambda i, e: frozenset((k(i, e[0]), k(i, e[1])))  # noqa: E731

    # 2. top, bottom and sides of every cell, as corner indices of its ring
    top, bot, sides = _orient_cells(c, k, edge)

    # 3. neighbours
    by_top = {}
    for i in range(n):
        by_top.setdefault(edge(i, top[i]), []).append(i)
    by_side = {}
    for i in range(n):
        for s in sides[i]:
            by_side.setdefault(edge(i, s), []).append(i)
    below = {}
    for i in range(n):
        js = by_top.get(edge(i, bot[i]), [])
        if len(js) > 1:
            raise ValueError(f"cell {i + 1}: more than one cell below it")
        if js:
            below[i] = js[0]
    above = {j: i for i, j in below.items()}
    side_nb = {i: [j for s in sides[i] for j in by_side[edge(i, s)] if j != i]
               for i in range(n)}

    # 4. the top row as one chain, then each column down
    top_cells = [i for i in range(n) if i not in above]
    top_set = set(top_cells)
    ends = [i for i in top_cells if len([j for j in side_nb[i] if j in top_set]) == 1]
    if len(top_cells) == 1:
        ends = top_cells
    if len(ends) != 2 and len(top_cells) > 1:
        raise ValueError(f"the {len(top_cells)} trench-row cells do not form one "
                         f"chain along strike ({len(ends)} chain ends)")
    chain = [ends[0]]
    while len(chain) < len(top_cells):
        nxt = [j for j in side_nb[chain[-1]] if j in top_set and j not in chain]
        if len(nxt) != 1:
            raise ValueError("the trench row breaks or branches along strike")
        chain.append(nxt[0])
    cols = []
    for t0 in chain:
        col = [t0]
        while col[-1] in below:
            col.append(below[col[-1]])
        cols.append(col)
    nrow = len(cols[0])
    if any(len(col) != nrow for col in cols):
        raise ValueError(f"columns of different lengths "
                         f"({sorted(set(len(col) for col in cols))} cells): not a "
                         f"rows x columns mesh")
    if nrow * len(cols) != n:
        raise ValueError(f"{n} cells but {len(cols)} columns x {nrow} rows")
    ncol = len(cols)

    # node lattice: cell (r, j)'s top-left corner is the top corner it shares
    # with column j-1 (for j = 0, the one it does not share with column 1)
    grid = np.full((nrow + 1, 3, ncol + 1), np.nan)

    def put(r, j, xyz):
        if not np.isnan(grid[r, 0, j]) and not np.allclose(grid[r, :, j], xyz, atol=1e-5):
            raise ValueError(f"node ({r}, {j}) gets two different positions: "
                             f"the cells do not share their corners")
        grid[r, :, j] = xyz

    for j, col in enumerate(cols):
        for r, i in enumerate(col):
            tp, bp = top[i], bot[i]
            # which top corner is on the column j+1 side?
            if j + 1 < ncol:
                right_keys = {k(cols[j + 1][r], p) for p in range(4)}
                t_right = [p for p in tp if k(i, p) in right_keys]
            else:
                left_keys = {k(cols[j - 1][r], p) for p in range(4)} if j > 0 else set()
                t_right = [p for p in tp if k(i, p) not in left_keys]
            if len(t_right) != 1:
                raise ValueError(f"cell {i + 1}: cannot tell its left and right "
                                 f"corners from its neighbours")
            tr = t_right[0]
            tl = tp[0] if tp[1] == tr else tp[1]
            # bottom corner on the right is the side partner of tr
            side_r = [s for s in sides[i] if tr in s][0]
            br = side_r[0] if side_r[1] == tr else side_r[1]
            bl = bp[0] if bp[1] == br else bp[1]
            put(r, j, c[i, tl])
            put(r, j + 1, c[i, tr])
            put(r + 1, j, c[i, bl])
            put(r + 1, j + 1, c[i, br])
    if np.isnan(grid).any():
        raise ValueError("the node lattice has holes")

    # 5. the fault must dip to the right of the strike (rptha's convention)
    def unit_xy(lon0, lat0, lon1, lat1):
        dx = (lon1 - lon0) * np.cos(np.radians((lat0 + lat1) / 2))
        dy = lat1 - lat0
        v = np.array([dx, dy])
        return v / np.linalg.norm(v)

    mid = ncol // 2
    s = unit_xy(grid[0, 0, mid], grid[0, 1, mid], grid[0, 0, mid + 1], grid[0, 1, mid + 1])
    dvec = unit_xy(grid[0, 0, mid], grid[0, 1, mid], grid[nrow, 0, mid], grid[nrow, 1, mid])
    right = s[0] * dvec[1] - s[1] * dvec[0] < 0  # cross(strike, dip) < 0: dip to the right
    if not right:
        grid = grid[:, :, ::-1].copy()
    log(f"      mesh file: {ncol} along-strike x {nrow} down-dip = {n} unit sources, "
        f"depth {grid[:, 2, :].min():.2f}-{grid[:, 2, :].max():.2f} km (file in "
        f"{depth_units}{', columns reversed so the fault dips to the right of the strike' if not right else ''})")
    return grid
