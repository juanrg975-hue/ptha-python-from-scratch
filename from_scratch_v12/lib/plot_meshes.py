"""Draw the from_scratch_v12 mesh, alone or against the official PTHA18 mesh.

Matches slab_vs_ptha/make_figures.py's form choices and palette so this
reads as one figure family with that study.

  fig_scratch_mesh_map   The scratch mesh's own cell POLYGONS on a map,
                         coloured by depth. Works with ONLY the .npy grid
                         from step 2 -- no official/PTHA18 data required, so
                         this always renders, including with --ptha=false on
                         a zone with no PTHA18 mesh to compare against at
                         all. This is what step 2 now calls automatically.

  fig_map_meshes         Both meshes' actual cell polygons side by side, same
                         depth colour scale -- the picture that answers "do
                         the two meshes occupy the same patch of the
                         interface" at a glance. Needs the official .nc
                         table; only runs when it is available.

  fig_profile_downdip    Depth vs. latitude along the arc, one line per mesh,
                         to show directly whether either mesh stops short of
                         the full arc. Also needs the official table.

Usage (standalone, for the comparison figures):
    .venv/Scripts/python.exe from_scratch_v12/lib/plot_meshes.py kermadectonga2 kermadectonga2_v3

Usage (scratch-only, no PTHA18 zone needed):
    .venv/Scripts/python.exe from_scratch_v12/lib/plot_meshes.py --scratch-only kermadectonga2_v3
"""

import argparse
import os

import numpy as np
import netCDF4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)  # from_scratch_v12/ (this file lives in lib/)
ROOT = os.path.dirname(PKG)
# v11: PTHA18's extracted files (rptha/, inputs/, official_ptha_data/) live in
# ptha18_logic_tree_test/; the package may sit there or one level down (V9/).
DATA_ROOT = next((r for r in (ROOT, os.path.dirname(ROOT))
                  if os.path.isdir(os.path.join(r, "rptha"))
                  or os.path.isdir(os.path.join(r, "official_ptha_data"))), ROOT)

# Same reference palette as slab_vs_ptha/make_figures.py (references/palette.md),
# so the two figure sets read as one family.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES_1 = "#2a78d6"   # blue   - official PTHA18
SERIES_2 = "#eb6834"   # orange - from_scratch_v12 (Berryman-derived)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "figure.dpi": 140,
})


def style(ax, grid_axis="both"):
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.tick_params(length=0)


# PTHA18's own unit-source grid, as published: one POLYGON per unit source,
# with its downdip_number / alongstrike_number. When this is present the
# official mesh is drawn from it exactly, instead of being reconstructed
# from centroid + length + width + strike (official_cell_polygons below).
OFFICIAL_GRID_SHP_DIR = os.path.join(
    PKG, "validation", "ptha18_reference", "unit_source_grid")


def official_cell_polygons_from_shapefile(zone, nc_path=None, shp_path=None):
    """The official mesh's TRUE cell polygons, from PTHA18's published
    unit-source-grid shapefile.

    Returns (polys, depth) with exactly the meaning and the ROW ORDER of
    official_cell_polygons(), so the two are interchangeable, or None if the
    shapefile for this zone is not available.

    Why the reindexing below is not optional
    ----------------------------------------
    The two files do not store their unit sources in the same order. The
    netCDF statistics table is DOWN-DIP major -- (dd, as) runs (1,1), (2,1),
    (3,1), (4,1), (1,2)... -- while the shapefile is ALONG-STRIKE major:
    (1,1), (1,2), (1,3)... Taking the shapefile's rows as they come would
    pair every polygon with another cell's depth (and, in
    fig_map_meshes_by_value, another cell's convergence), which looks
    plausible and is wrong. So the polygons are indexed by their
    (downdip_number, alongstrike_number) and pulled out in the netCDF's own
    order.
    """
    import geopandas as gpd

    shp_path = shp_path or os.path.join(OFFICIAL_GRID_SHP_DIR, f"{zone}.shp")
    if not os.path.exists(shp_path):
        return None
    nc_path = nc_path or os.path.join(DATA_ROOT, "inputs", "geometry",
                                      f"unit_source_statistics_{zone}.nc")
    if not os.path.exists(nc_path):
        return None

    with netCDF4.Dataset(nc_path) as f:
        depth = np.asarray(f["depth"][:], dtype=float)
        dd = np.asarray(f["downdip_number"][:], dtype=int)
        al = np.asarray(f["alongstrike_number"][:], dtype=int)

    gdf = gpd.read_file(shp_path)
    by_key = {(int(r.dwndp_n), int(r.alngst_)): r.geometry
              for r in gdf.itertuples()}
    if len(by_key) != len(gdf) or len(gdf) != depth.size:
        return None

    polys = []
    for k in range(depth.size):
        geom = by_key.get((int(dd[k]), int(al[k])))
        if geom is None:
            return None
        # Exterior ring, dropping the repeated closing vertex: PolyCollection
        # closes each polygon itself.
        x, y = geom.exterior.coords.xy
        polys.append(np.column_stack([np.asarray(x)[:-1], np.asarray(y)[:-1]]))
    return polys, depth


def official_cell_polygons_best(zone, path=None):
    """The official mesh's cells: the published polygons when the shapefile
    is there, otherwise the rectangle reconstruction. Returns
    (polys, depth, source) where source is "shapefile" or "reconstructed"."""
    got = official_cell_polygons_from_shapefile(zone)
    if got is not None:
        return got[0], got[1], "shapefile"
    polys, depth = official_cell_polygons(zone, path=path)
    return polys, depth, "reconstructed"


def official_cell_polygons(zone, path=None):
    """Approximate each official unit source as a lon/lat rectangle.

    Fallback for a zone with no published unit-source-grid shapefile; where
    one exists, official_cell_polygons_best() uses it instead and this is
    not called.

    The published table gives a centroid, length (along-strike), width
    (down-dip) and strike, not the corner coordinates directly -- so each
    cell is reconstructed as the rectangle of that length/width centred on
    the centroid and rotated to strike. This is an approximation for
    plotting only; it is not used anywhere in the rate calculation.
    """
    path = path or os.path.join(DATA_ROOT, "inputs", "geometry",
                                f"unit_source_statistics_{zone}.nc")
    with netCDF4.Dataset(path) as f:
        lon = np.asarray(f["lon_c"][:], dtype=float)
        lat = np.asarray(f["lat_c"][:], dtype=float)
        depth = np.asarray(f["depth"][:], dtype=float)
        length = np.asarray(f["length"][:], dtype=float)
        width = np.asarray(f["width"][:], dtype=float)
        strike = np.asarray(f["strike"][:], dtype=float)
        dip = np.asarray(f["dip"][:], dtype=float)

    # Surface (map-view) width of the cell: the down-dip width projects onto
    # the horizontal by cos(dip).
    surf_width = width * np.cos(np.radians(dip))

    km_per_deg_lat = 111.32
    polys = []
    for i in range(lon.size):
        km_per_deg_lon = km_per_deg_lat * np.cos(np.radians(lat[i]))
        half_l, half_w = length[i] / 2.0, surf_width[i] / 2.0
        th = np.radians(strike[i])
        # Local rectangle corners (along-strike x, cross-strike y), in km,
        # then rotated by strike (bearing from north, clockwise) and
        # converted to degrees.
        corners_km = np.array([
            [-half_l, -half_w], [half_l, -half_w],
            [half_l, half_w], [-half_l, half_w]])
        # Rotation: strike is a bearing (0=N, 90=E), and the along-strike
        # axis points along that bearing.
        rot = np.array([[np.sin(th), np.cos(th)],
                        [np.cos(th), -np.sin(th)]])
        xy = corners_km @ rot.T
        dlon = xy[:, 0] / km_per_deg_lon
        dlat = xy[:, 1] / km_per_deg_lat
        polys.append(np.column_stack([lon[i] + dlon, lat[i] + dlat]))
    return polys, depth


def scratch_cell_polygons(grid):
    """Cell polygons straight from a unit_source_grid array.

    `grid` has shape (n_downdip+1, 3, n_alongstrike+1): grid[j, :, i] is the
    (lon, lat, depth) of down-dip line j at along-strike position i. Each
    cell (i, j) is the quadrilateral of its four surrounding grid points --
    exactly what the discretiser built, no reconstruction needed.
    """
    n_down = grid.shape[0] - 1
    n_along = grid.shape[2] - 1
    polys, depths = [], []
    for i in range(n_along):
        for j in range(n_down):
            corners = np.array([grid[j, :2, i], grid[j, :2, i + 1],
                                grid[j + 1, :2, i + 1], grid[j + 1, :2, i]])
            polys.append(corners)
            depths.append(float(np.mean([grid[j, 2, i], grid[j, 2, i + 1],
                                         grid[j + 1, 2, i + 1],
                                         grid[j + 1, 2, i]])))
    return polys, np.array(depths)


def fig_scratch_mesh_map(scratch_polys, scratch_depth, zone_label, n_downdip,
                        n_alongstrike, out_dir, out_name="scratch_mesh_map.png"):
    """The scratch mesh alone: its own cell polygons on a map, coloured by
    depth. Needs nothing but the .npy grid from step 2 -- no official/PTHA18
    data of any kind -- so this always renders, on any zone, with or
    without --ptha, with or without an official PTHA18 mesh to compare
    against. This is what step 2 calls automatically after building the
    grid.
    """
    from matplotlib.collections import PolyCollection

    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    pc = PolyCollection(scratch_polys, array=scratch_depth, cmap="Oranges",
                        edgecolor=INK_2, linewidth=0.3, zorder=3)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_xlabel("longitude (deg E)")
    ax.set_ylabel("latitude (deg N)")
    style(ax)

    cb = fig.colorbar(pc, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label("depth to interface (km, positive down)", color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=MUTED)

    fig.suptitle(f"{zone_label}: {len(scratch_polys)} unit sources "
                 f"({n_alongstrike} along strike x {n_downdip} down dip)",
                 x=0.02, ha="left", fontsize=11.5, fontweight="semibold")

    out = os.path.join(out_dir, out_name)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_scratch_mesh_map_by_value(scratch_polys, values, zone_label, n_downdip,
                                  n_alongstrike, out_dir, value_label,
                                  cmap="viridis", out_name="scratch_mesh_map_value.png"):
    """Same cell-polygon map as fig_scratch_mesh_map, coloured by an
    arbitrary per-cell value instead of depth -- e.g. step 3's per-cell
    Bird (2003) convergent slip. `values` must be in the same order as
    `scratch_polys` (both come from iterating the grid along-strike outer,
    down-dip inner -- see scratch_cell_polygons and
    discretized_source_approximate_summary_statistics, which share that
    loop order by construction).
    """
    from matplotlib.collections import PolyCollection

    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    pc = PolyCollection(scratch_polys, array=np.asarray(values), cmap=cmap,
                        edgecolor=INK_2, linewidth=0.3, zorder=3)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_xlabel("longitude (deg E)")
    ax.set_ylabel("latitude (deg N)")
    style(ax)

    cb = fig.colorbar(pc, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label(value_label, color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=MUTED)

    fig.suptitle(f"{zone_label}: {len(scratch_polys)} unit sources "
                 f"({n_alongstrike} along strike x {n_downdip} down dip)",
                 x=0.02, ha="left", fontsize=11.5, fontweight="semibold")

    out = os.path.join(out_dir, out_name)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _match_lon_convention(off_polys, scratch_polys):
    """Shift the official polygons by +-360 deg if needed so both meshes use
    the same longitude convention (0-360 vs -180/180).

    Both conventions are individually valid; the official netCDF table
    (lon_c) and this package's own unit_source_grid.npy (unwrapped in step 1
    to keep a curved arc continuous, see unwrap_antimeridian) do not always
    agree on which one they use. With sharex=True (fig_map_meshes), a mismatch
    stretches the shared axis across the full gap between them and both
    meshes collapse to a sliver -- only cosmetic, the meshes themselves are
    each correct in their own polygons.
    """
    off_mean = np.mean([p[:, 0].mean() for p in off_polys])
    scratch_mean = np.mean([p[:, 0].mean() for p in scratch_polys])
    shift = 360.0 * round((scratch_mean - off_mean) / 360.0)
    if shift == 0.0:
        return off_polys
    return [p + np.array([shift, 0.0]) for p in off_polys]


def fig_map_meshes(off_polys, off_depth, scratch_polys, scratch_depth, off_zone, scratch_folder,
                   out_dir, scratch_label="from_scratch_v12 (Berryman-derived)"):
    """Both meshes' actual cell polygons, on the same map and depth scale."""
    from matplotlib.collections import PolyCollection

    off_polys = _match_lon_convention(off_polys, scratch_polys)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.6), sharex=True, sharey=True)

    vmax = float(max(off_depth.max(), scratch_depth.max()))
    panels = [("Official PTHA18 mesh", off_polys, off_depth, "Blues"),
              (scratch_label, scratch_polys, scratch_depth, "Oranges")]

    for ax, (title, polys, depth, cmap) in zip(axes, panels):
        pc = PolyCollection(polys, array=depth, cmap=cmap,
                            edgecolor=INK_2, linewidth=0.3, zorder=3)
        pc.set_clim(0.0, vmax)
        ax.add_collection(pc)
        ax.autoscale_view()
        ax.set_title(title, loc="left")
        ax.set_xlabel("longitude (deg E)")
        style(ax)

    axes[0].set_ylabel("latitude (deg N)")
    cb = fig.colorbar(pc, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("depth to interface (km, positive down)", color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=MUTED)

    fig.suptitle(f"{off_zone}: {len(off_polys)} official cells vs. "
                 f"{len(scratch_polys)} from-scratch cells",
                 x=0.005, ha="left", fontsize=11.5, fontweight="semibold")

    out = os.path.join(out_dir, "mesh_comparison_map.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_map_meshes_by_value(off_polys, off_values, scratch_polys, scratch_values,
                            off_zone, scratch_folder, value_label, out_dir,
                            cmap="viridis", out_name="mesh_comparison_map_value.png"):
    """Both meshes' cell polygons side by side, coloured by an arbitrary
    per-cell value (e.g. Bird convergence) on ONE shared colour scale --
    same idea as fig_map_meshes, generalised past depth.
    """
    from matplotlib.collections import PolyCollection

    off_polys = _match_lon_convention(off_polys, scratch_polys)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.6), sharex=True, sharey=True)

    off_values = np.asarray(off_values)
    scratch_values = np.asarray(scratch_values)
    vmin = float(min(off_values.min(), scratch_values.min()))
    vmax = float(max(off_values.max(), scratch_values.max()))
    panels = [("Official PTHA18 mesh", off_polys, off_values),
              (f"from_scratch_v12 ({scratch_folder})", scratch_polys, scratch_values)]

    for ax, (title, polys, values) in zip(axes, panels):
        pc = PolyCollection(polys, array=values, cmap=cmap,
                            edgecolor=INK_2, linewidth=0.3, zorder=3)
        pc.set_clim(vmin, vmax)
        ax.add_collection(pc)
        ax.autoscale_view()
        ax.set_title(title, loc="left")
        ax.set_xlabel("longitude (deg E)")
        style(ax)

    axes[0].set_ylabel("latitude (deg N)")
    cb = fig.colorbar(pc, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label(value_label, color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=MUTED)

    fig.suptitle(f"{off_zone}: {len(off_polys)} official cells vs. "
                 f"{len(scratch_polys)} from-scratch cells",
                 x=0.005, ha="left", fontsize=11.5, fontweight="semibold")

    out = os.path.join(out_dir, out_name)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# One colour per segment, in along-strike order, the same in every panel so
# "the first segment" reads the same colour on both meshes.
SEGMENT_COLOURS = ["#2a78d6", "#eb6834", "#2e9e5b", "#9b59b6", "#d4a017",
                   "#17a2b8", "#c0392b", "#7f8c8d"]


def fig_map_segments(panels, out_dir, out_name="segments_map.png"):
    """v9 (LEVEL 0). Each mesh with its cells coloured by the segment they
    belong to, one panel per mesh, on shared axes.

    panels: list of (title, polys, alongstrike, segments), where
    `alongstrike` is each polygon's 1-based column and `segments` is a list
    of (name, [first_column, last_column]) in along-strike order. Each
    segment is labelled at the centre of its cells.
    """
    from matplotlib.collections import PolyCollection

    ref = panels[-1][1]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.7 * len(panels), 5.8),
                             sharex=True, sharey=True, squeeze=False)
    for ax, (title, polys, al, segs) in zip(axes[0], panels):
        polys = _match_lon_convention(polys, ref) if polys is not ref else polys
        al = np.asarray(al, dtype=int)
        colours = np.array(["#d9d9d9"] * len(polys), dtype=object)
        for k, (name, (a, b)) in enumerate(segs):
            colours[(al >= a) & (al <= b)] = SEGMENT_COLOURS[k % len(SEGMENT_COLOURS)]
        ax.add_collection(PolyCollection(polys, facecolors=list(colours),
                                         edgecolor="white", linewidth=0.35,
                                         zorder=3))
        for k, (name, (a, b)) in enumerate(segs):
            pts = np.vstack([polys[i] for i in np.where((al >= a) & (al <= b))[0]])
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            ax.text(cx, cy, f"{name}\ncols {a}-{b}", ha="center", va="center",
                    fontsize=8.5, fontweight="semibold", color=INK, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                              alpha=0.85))
        ax.autoscale_view()
        ax.set_title(title, loc="left")
        ax.set_xlabel("longitude (deg E)")
        style(ax)
    axes[0][0].set_ylabel("latitude (deg N)")
    out = os.path.join(out_dir, out_name)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def write_mesh_csv(polys, depth, path, downdip=None, alongstrike=None,
                   extra=None, extra_name=None):
    """One row per unit source: lon, lat, depth -- the mesh as plain text.

    lon/lat are the cell CENTROID (the mean of its corners) and depth is the
    cell-centroid depth in km below the sea floor datum, positive down: the
    same three numbers the maps in the report are drawn from, so a row here
    is one coloured cell there.

    downdip / alongstrike, when given, are written first so a row can be
    located in the mesh without recomputing anything. `extra` adds one more
    column under the name `extra_name` (step 9 uses it for the per-cell Bird
    convergence).

    Written for BOTH meshes when both exist -- this run's and, from step 8,
    PTHA18's own -- so the two can be diffed directly rather than read off
    two pictures.
    """
    depth = np.asarray(depth, dtype=float)
    cols = ["lon", "lat", "depth_km"]
    if downdip is not None:
        cols = ["downdip_number", "alongstrike_number"] + cols
    if extra is not None:
        cols.append(extra_name or "value")

    lines = [",".join(cols)]
    for i, poly in enumerate(polys):
        row = []
        if downdip is not None:
            row += [str(int(downdip[i])), str(int(alongstrike[i]))]
        row += [f"{np.asarray(poly)[:, 0].mean():.6f}",
                f"{np.asarray(poly)[:, 1].mean():.6f}",
                f"{depth[i]:.4f}"]
        if extra is not None:
            row.append(f"{float(np.asarray(extra)[i]):.6f}")
        lines.append(",".join(row))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def scratch_cell_indices(grid):
    """(downdip_number, alongstrike_number) per cell, 1-based, in exactly the
    order scratch_cell_polygons() returns its polygons (along-strike outer,
    down-dip inner)."""
    n_down = grid.shape[0] - 1
    n_along = grid.shape[2] - 1
    dd, al = [], []
    for i in range(n_along):
        for j in range(n_down):
            dd.append(j + 1)
            al.append(i + 1)
    return np.array(dd), np.array(al)


def fig_profile_downdip_scratch(scratch_grid, zone_label, out_dir,
                                out_name="mesh_profile.png"):
    """Depth vs. latitude along the arc for THIS run's mesh alone.

    The same picture as fig_profile_downdip, minus the official series, so it
    renders on any run -- with --ptha false, or before step 8 has been run.
    Reading it: each line is one down-dip row, and the span on the x axis is
    the arc the mesh actually covers. Depths are CELL CENTROIDS (the mean of
    each cell's four corners), the convention the official table's own
    `depth` column uses, so this panel and the official one are directly
    comparable when both exist.
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    n_down = scratch_grid.shape[0] - 1
    for j in range(n_down):
        lat_row = 0.25 * (scratch_grid[j, 1, :-1] + scratch_grid[j, 1, 1:]
                          + scratch_grid[j + 1, 1, :-1] + scratch_grid[j + 1, 1, 1:])
        dep_row = 0.25 * (scratch_grid[j, 2, :-1] + scratch_grid[j, 2, 1:]
                          + scratch_grid[j + 1, 2, :-1] + scratch_grid[j + 1, 2, 1:])
        order = np.argsort(lat_row)
        ax.plot(lat_row[order], dep_row[order], color=SERIES_2, linewidth=1.6,
                marker="s", markersize=3, zorder=3,
                label=f"down-dip row {j + 1}" if n_down <= 8 else
                      ("this run" if j == 0 else None))

    ax.invert_yaxis()
    ax.set_xlabel("latitude (deg N)")
    ax.set_ylabel("depth to interface (km, positive down)")
    ax.set_title(f"{zone_label}: depth along the arc, one line per down-dip row",
                 loc="left")
    ax.legend(frameon=False, loc="upper right", ncol=2 if n_down > 4 else 1)
    style(ax)

    out = os.path.join(out_dir, out_name)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_profile_downdip(off_polys, off_depth, off_zone_dd, scratch_grid, out_dir):
    """Depth vs. down-dip row, one series per mesh.

    Plotted against LATITUDE on the x-axis (not row index), because that is
    what distinguishes "this mesh covers the whole arc" from "this mesh
    stops short of Tonga" -- the exact failure mode earlier v2 attempts had.

    Both series are CELL-CENTROID depths, not grid-line (row-edge) depths.
    An earlier version of this plot compared the official centroids against
    v2's down-dip grid LINES -- edge 0, edge 1, edge 2, edge 3 -- which sit
    at the top and bottom of each row rather than its middle, so v2's curves
    were offset deeper by construction, looking like a large mismatch
    (+several km) that was mostly a plotting artefact, not a geometry one.
    Averaging each cell's four corner depths puts both meshes on the same
    convention: the official table's own `depth` column is exactly this
    same average.
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    off_lat = np.array([p[:, 1].mean() for p in off_polys])
    for row in sorted(set(off_zone_dd.tolist())):
        m = off_zone_dd == row
        idx = np.where(m)[0]
        idx = idx[np.argsort(off_lat[idx])]
        ax.plot(off_lat[idx], off_depth[idx], color=SERIES_1, linewidth=1.6,
                marker="o", markersize=3,
                label="official PTHA18" if row == sorted(set(off_zone_dd.tolist()))[0]
                else None, zorder=3)

    n_down = scratch_grid.shape[0] - 1
    n_along = scratch_grid.shape[2] - 1
    for j in range(n_down):
        # Cell-centroid depth: mean of the 4 corners bounding row j, at each
        # along-strike position -- matches the official table's convention.
        lat_row = 0.25 * (scratch_grid[j, 1, :-1] + scratch_grid[j, 1, 1:]
                         + scratch_grid[j + 1, 1, :-1] + scratch_grid[j + 1, 1, 1:])
        dep_row = 0.25 * (scratch_grid[j, 2, :-1] + scratch_grid[j, 2, 1:]
                         + scratch_grid[j + 1, 2, :-1] + scratch_grid[j + 1, 2, 1:])
        order2 = np.argsort(lat_row)
        ax.plot(lat_row[order2], dep_row[order2], color=SERIES_2, linewidth=1.6,
                marker="s", markersize=3,
                label="from_scratch_v12" if j == 0 else None, zorder=3)

    ax.invert_yaxis()
    ax.set_xlabel("latitude (deg N)")
    ax.set_ylabel("depth to interface (km, positive down)")
    ax.set_title("Depth along the arc, by mesh: does either stop short?",
                 loc="left")
    ax.legend(frameon=False, loc="upper right")
    style(ax)

    out = os.path.join(out_dir, "mesh_comparison_profile.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("zone", nargs="?", default=None,
                    help="official PTHA18 zone name, e.g. kermadectonga2 -- "
                         "omit with --scratch-only")
    ap.add_argument("scratch_folder", help="from_scratch_v12 example folder, "
                                      "e.g. kermadectonga2_v3")
    ap.add_argument("--scratch-only", action="store_true",
                    help="draw only the scratch mesh, never touch PTHA18's "
                         "official table -- the mode this package uses with "
                         "--ptha=false")
    args = ap.parse_args()
    if not args.scratch_only and args.zone is None:
        ap.error("zone is required unless --scratch-only is given")

    scratch_dir = os.path.join(ROOT, args.scratch_folder)
    scratch_npy = os.path.join(scratch_dir, "data", "slab2", "unit_source_grid.npy")
    if not os.path.exists(scratch_npy):
        raise SystemExit(f"missing {scratch_npy}\nRun steps 1-2 in {args.scratch_folder}/ first.")

    print(f"from_scratch_v12 mesh: {args.scratch_folder}")
    scratch_grid = np.load(scratch_npy)
    scratch_polys, scratch_depth = scratch_cell_polygons(scratch_grid)
    n_downdip = scratch_grid.shape[0] - 1
    n_alongstrike = scratch_grid.shape[2] - 1
    print(f"  {len(scratch_polys)} cells "
          f"({n_alongstrike} along strike x {n_downdip} down dip)")

    out_dir = scratch_dir
    p0 = fig_scratch_mesh_map(scratch_polys, scratch_depth, args.scratch_folder,
                             n_downdip, n_alongstrike, out_dir)
    print(f"  wrote {os.path.relpath(p0, ROOT)}")

    if args.scratch_only:
        return

    nc_path = os.path.join(DATA_ROOT, "inputs", "geometry",
                           f"unit_source_statistics_{args.zone}.nc")
    if not os.path.exists(nc_path):
        print(f"\n  no official table at {os.path.relpath(nc_path, ROOT)} -- "
              f"skipping the official-vs-scratch comparison figures. "
              f"Download it with official_ptha_data/fetch_official_inputs.py "
              f"{args.zone}, or pass --scratch-only to skip this step "
              f"entirely.")
        return

    print(f"\nOfficial mesh: {args.zone}")
    off_polys, off_depth, off_src = official_cell_polygons_best(args.zone)
    print(f"  cells from: {off_src}")
    with netCDF4.Dataset(nc_path) as f:
        off_dd = np.asarray(f["downdip_number"][:], dtype=int)
    print(f"  {len(off_polys)} cells")

    p1 = fig_map_meshes(off_polys, off_depth, scratch_polys, scratch_depth, args.zone,
                        args.scratch_folder, out_dir)
    print(f"  wrote {os.path.relpath(p1, ROOT)}")
    p2 = fig_profile_downdip(off_polys, off_depth, off_dd, scratch_grid, out_dir)
    print(f"  wrote {os.path.relpath(p2, ROOT)}")


if __name__ == "__main__":
    main()
