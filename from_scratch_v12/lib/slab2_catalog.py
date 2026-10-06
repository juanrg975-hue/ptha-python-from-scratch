"""The 27 SLAB2.0 regions on ScienceBase, verified against the live API.

Each entry is (ScienceBase item id, region title, shapefile prefix). The
prefix is the first part of that region's <prefix>_shapefiles.zip and
<prefix>_depth.shp -- e.g. Kermadec is item 5aa318e1e4b0b1c392ea3f10 with
prefix "ker", so its files are ker_shapefiles.zip / ker_depth.shp.

This list was built by querying the ScienceBase API directly:
    https://www.sciencebase.gov/catalog/items?parentId=5aa1b00ee4b0b1c392e86467
      (the SLAB2 parent item, DOI 10.5066/F7PV6JNV)
then, per child item:
    https://www.sciencebase.gov/catalog/item/<id>?format=json&fields=files,title
      (to read the file named "*_shapefiles.zip" and its prefix)

Every row here was confirmed to exist; none is guessed from a naming pattern.
generate.py resolves the actual download URL (which embeds a content hash
ScienceBase assigns per file, e.g. "?f=__disk__93%2F43%2Fef%2F...") from the
item id at generation time via the same API call, rather than hardcoding it,
so this catalog does not go stale when files are re-uploaded.
"""

# (scibase_item_id, title, shapefile_prefix)
REGIONS = [
    ("5aa2c535e4b0b1c392ea3ca2", "Alaska", "alu"),
    ("5aa31058e4b0b1c392ea3e63", "Calabria", "cal"),
    ("5aa311dbe4b0b1c392ea3ef2", "Caribbean", "car"),
    ("5aa312cde4b0b1c392ea3ef5", "Cascadia", "cas"),
    ("5aa31127e4b0b1c392ea3e68", "Central America", "cam"),
    ("5aa314dae4b0b1c392ea3efe", "Cotabato", "cot"),
    ("5aa3156fe4b0b1c392ea3f01", "Halmahera", "hal"),
    ("5aa31604e4b0b1c392ea3f04", "Hellenic Arc", "hel"),
    ("5aa316fae4b0b1c392ea3f07", "Himalaya", "him"),
    ("5aa3177ae4b0b1c392ea3f0a", "Hindu Kush", "hin"),
    ("5aa3185ee4b0b1c392ea3f0d", "Izu-Bonin", "izu"),
    ("5aa4060de4b0b1c392eaaee2", "Kamchatka-Kuril Islands-Japan", "kur"),
    ("5aa318e1e4b0b1c392ea3f10", "Kermadec", "ker"),
    ("5aa406f1e4b0b1c392eaaee5", "Makran", "mak"),
    ("5aa4076fe4b0b1c392eaaee8", "Manila Trench", "man"),
    ("5aa40800e4b0b1c392eaaeeb", "Muertos Trough", "mue"),
    ("5aa413f2e4b0b1c392eaaf2a", "New Guinea", "png"),
    ("5aa40985e4b0b1c392eaaeee", "Pamir", "pam"),
    ("5aa40a33e4b0b1c392eaaef4", "Philippines", "phi"),
    ("5aa412b2e4b0b1c392eaaf27", "Puysegur", "puy"),
    ("5aa40aafe4b0b1c392eaaefa", "Ryukyu", "ryu"),
    ("5aa41674e4b0b1c392eaaf31", "Scotia Sea", "sco"),
    ("5aa41721e4b0b1c392eaaf35", "Solomon Islands", "sol"),
    ("5aa41473e4b0b1c392eaaf2d", "South America", "sam"),
    ("5aa417cbe4b0b1c392eaaf38", "Sulawesi", "sul"),
    ("5aa41834e4b0b1c392eaaf3b", "Sumatra-Java", "sum"),
    ("5aa4189ee4b0b1c392eaaf3d", "Vanuatu", "van"),
]


def by_id(item_id):
    for iid, title, prefix in REGIONS:
        if iid == item_id:
            return {"id": iid, "title": title, "prefix": prefix}
    return None


def find_region(query):
    """Match `query` against region titles and prefixes.

    Case-insensitive, spaces/hyphens ignored (so "south america" and
    "southamerica" both hit "South America"). Returns a list of matching
    region dicts -- 0, 1, or more than 1 (e.g. "island" would hit several).
    A single hit is the common case; the caller decides what to do with 0 or
    many.
    """
    squashed_query = query.lower().replace(" ", "").replace("-", "").replace("_", "")
    exact, hits = [], []
    for iid, title, prefix in REGIONS:
        squashed_title = title.lower().replace(" ", "").replace("-", "")
        r = {"id": iid, "title": title, "prefix": prefix}
        if squashed_query in (prefix.lower(), squashed_title):
            exact.append(r)
        elif squashed_query in squashed_title or squashed_title in squashed_query:
            hits.append(r)
    # an exact prefix or title wins: "car" is Caribbean even though it is
    # also inside "hellenicarc", and "cot" is Cotabato, not "scotiasea"
    return exact or hits


def guess_ptha18_zones(region_title, zone_names):
    """Substring match, either direction, case-insensitive, on a squashed
    (no spaces/hyphens) version of the region title. Best-effort only."""
    squashed = region_title.lower().replace(" ", "").replace("-", "")
    hits = []
    for z in zone_names:
        zl = z.lower()
        if zl in squashed or squashed in zl or zl[:5] in squashed:
            hits.append(z)
    return hits
