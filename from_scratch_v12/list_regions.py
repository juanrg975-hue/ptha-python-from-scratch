"""List the SLAB2.0/SLAB1.0 regions available to build a from-scratch example.

Browse this if you are not sure what to type, or if generate.py tells you a
name is ambiguous or not found. Most of the time you can skip straight to
generate.py with just the name you want (e.g. "cascadia") -- it resolves the
region and the PTHA18 zone name itself.

A SLAB2 region and a PTHA18 source zone are two different catalogues that
happen to describe the same faults under different names (SLAB2's "Kermadec"
is PTHA18's "kermadectonga2", for instance) -- the "PTHA18 zone guess" column
below is a best-effort substring match, not a guarantee.

Which SLAB product a given PTHA18 zone actually uses is decided by
generate.py/official_geometry_params.py's slab_product_for_zone(): a zone name
ending in "2" used SLAB2.0, every other zone used SLAB1.0 (ReportPTHA.pdf
p.11-12). SLAB1.0 has no ScienceBase catalog of its own -- it reuses the same
three-letter region codes as SLAB2.0 (see ZONE_TO_SLAB1_REGION), so the second
table below just re-groups the same 27 regions by their SLAB1.0 zones instead.

Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe from_scratch_v12/list_regions.py
"""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
# v12: rptha/ next to the package, or one level up (package in V9/)
_RPTHA_ROOT = next((r for r in (ROOT, os.path.dirname(ROOT))
                    if os.path.isdir(os.path.join(r, "rptha"))), ROOT)
SZP = os.path.join(_RPTHA_ROOT, "rptha", "R", "examples", "austptha_template", "DATA",
                   "SOURCEZONE_PARAMETERS", "sourcezone_parameters.csv")

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lib"))  # helper modules
from slab2_catalog import REGIONS, guess_ptha18_zones  # noqa: E402
from official_geometry_params import slab_product_for_zone  # noqa: E402


def unsegmented_zone_names():
    """Every sourcename in sourcezone_parameters.csv with no segment_name."""
    names = []
    with open(SZP, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("segment_name", "").strip():
                names.append(row["sourcename"].strip())
    return names


def main():
    zone_names = unsegmented_zone_names()

    print("=" * 78)
    print("SLAB2.0 regions (from ScienceBase, DOI 10.5066/F7PV6JNV)")
    print("=" * 78)
    print(f"{'ScienceBase item id':<26} {'Region':<32} {'PTHA18 zone guess'}")
    print("-" * 78)
    for item_id, title, prefix in REGIONS:
        guesses = guess_ptha18_zones(title, zone_names)
        guess_txt = ", ".join(guesses) if guesses else "(none found)"
        print(f"{item_id:<26} {title:<32} {guess_txt}")

    print()
    print("=" * 78)
    print("SLAB1.0 regions (same 3-letter codes as SLAB2.0, reused per-zone --")
    print("see ZONE_TO_SLAB1_REGION; grids fetched from earthquake.usgs.gov)")
    print("=" * 78)
    print(f"{'Prefix':<8} {'Region':<32} {'PTHA18 zones using SLAB1.0'}")
    print("-" * 78)
    for item_id, title, prefix in REGIONS:
        guesses = guess_ptha18_zones(title, zone_names)
        slab1_guesses = [z for z in guesses if slab_product_for_zone(z) == "SLAB1.0"]
        guess_txt = ", ".join(slab1_guesses) if slab1_guesses else "(none found)"
        print(f"{prefix:<8} {title:<32} {guess_txt}")

    print()
    print("Usually all you need is:")
    print("  .venv/Scripts/python.exe from_scratch_v12/generate.py cascadia")
    print()
    print("If that name is ambiguous, not found, or picks the wrong PTHA18")
    print("zone, be explicit:")
    print("  .venv/Scripts/python.exe from_scratch_v12/generate.py "
         "cascadia --zone cascadia --folder cascadia")
    print()
    print("The SLAB product (SLAB1.0 vs SLAB2.0) is chosen automatically from")
    print("the PTHA18 zone name (a zone ending in '2' used SLAB2.0) -- use")
    print("--ptha false to avoid PTHA18's own mesh row-count/edges, not to")
    print("pick a SLAB product.")


if __name__ == "__main__":
    sys.exit(main())
