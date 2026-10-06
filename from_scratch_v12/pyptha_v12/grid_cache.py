"""One mesh per run: step 2 builds it, steps 7 and 9 reuse it (new in v8).

v4-v7 built the unit-source mesh three times per run: in step 2, again in
step 7 (run_logic_tree.py's build_grid, from the input JSON) and again in
step 9 (the report). Each rebuild re-ran the Levenberg-Marquardt optimiser,
and in v5-v7 that optimiser had a wall-clock timeout with a random restart,
so the mesh used for the rates could differ from the one step 2 checked,
depending on machine speed.

v8's optimiser is deterministic (see contour_discretisation._nls_lm), so a
rebuild gives the same mesh; this module just avoids paying for it twice.
Step 2 saves the grid next to a small JSON "fingerprint": a SHA-256 of the
contour shapefile's bytes plus every meshing parameter. Whoever needs the
mesh again asks for it with the same inputs; if the fingerprint matches, the
saved grid is returned, otherwise the mesh is rebuilt (for example after
desired_unit_source_width was edited in the input JSON).
"""

import hashlib
import json
import os

import numpy as np

# Bump when anything that changes the mesh for the same inputs changes.
MESH_CODE_VERSION = "v8.0"


def _shapefile_digest(shp_path):
    h = hashlib.sha256()
    base = os.path.splitext(shp_path)[0]
    for ext in (".shp", ".dbf", ".shx"):
        p = base + ext
        if os.path.exists(p):
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def fingerprint(shp_path, **params):
    """Everything that determines the mesh, as a JSON-able dict."""
    fp = {"contours_sha256": _shapefile_digest(shp_path),
          "mesh_code_version": MESH_CODE_VERSION}
    for k in sorted(params):
        v = params[k]
        fp[k] = None if v is None else (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
    return fp


def meta_path(npy_path):
    return os.path.splitext(npy_path)[0] + ".meta.json"


def save(npy_path, grid, fp):
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, grid)
    with open(meta_path(npy_path), "w", encoding="utf-8") as fh:
        json.dump(fp, fh, indent=1)


def load_if_match(npy_path, fp):
    """The saved grid if its fingerprint equals `fp`, else None."""
    mp = meta_path(npy_path)
    if not (os.path.exists(npy_path) and os.path.exists(mp)):
        return None
    try:
        with open(mp, encoding="utf-8") as fh:
            saved = json.load(fh)
    except (OSError, ValueError):
        return None
    if saved != json.loads(json.dumps(fp)):
        return None
    return np.load(npy_path)
