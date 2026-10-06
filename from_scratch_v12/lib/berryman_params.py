"""Literature inputs (coupling, b-value range, mw_max, seismogenic cutoff)
read from the actual primary source PTHA18 cites, instead of from PTHA18's
own already-computed table.

Why this module exists
-----------------------
from_scratch_v2's step6_write_input.py reads coupling (cmin/cpref/cmax),
b_anchor (bmin/bpref/bmax) and mw_max_observed from
    rptha/.../DATA/SOURCEZONE_PARAMETERS/sourcezone_parameters.csv
which is PTHA18's OWN internal table -- already computed, not a primary
source. For the three zones tested so far that is a defensible literature
lookup (the CSV says outright it took these numbers from Berryman et al.
2015). But it means v2 cannot be reused unmodified for a zone PTHA18 never
covered, because there is no sourcezone_parameters.csv row for it.

This module reads the same three numbers from the actual cited paper:

    Berryman, K.; Wallace, L.; Hayes, G.; Bird, P.; Wang, K.; Basili, R.;
    Lay, T.; Pagani, M.; Stein, R.; Sagiya, T.; Rubin, C.; Barreintos, S.;
    Kreemer, C.; Litchfield, N.; Stirling, M.; Gledhill, K.; Haller, K.;
    Costa, C. (2015) The GEM Faulted Earth Subduction Interface
    Characterisation Project, Version 2.0, April 2015.
    GNS Science Miscellaneous Series 80. 34 p. + Appendices.

Table 3.1 of that report (verified visually against the rendered PDF pages,
not just pdftotext output -- see BERRYMAN_TABLE_SOURCE below) gives, per
subduction-zone segment: dip, trench depth, up-dip depth (pref/min/max),
down-dip depth (pref/min/max, all in km BELOW SEA LEVEL, not below trench),
down-dip width, and coupling coefficient (min/max in Table 3.1's coupling
appendix table, pref in the geometry table's last column). Mmax and b-value
ranges are in a separate coupling/Mmax table (also verified visually).

The seismogenic-cutoff derivation
----------------------------------
ReportPTHA.pdf (the PTHA18 report itself) says: "The [maximum depth below
the trench] was mainly estimated using the database of Berryman et al.
(2015) by subtracting the trench depth below mean sea level (MSL) from
their largest recommended seismogenic depth limit, and rounding up to a
multiple of 5 km."

"Largest recommended" = Berryman's down-dip-depth MAX column (not pref).
Verified against all three zones this package has official cutoffs for:
  kermadectonga2 (dominant segment: Tonga)            45-8=37 -> 40 km  (matches official 40)
  puysegur2      (single segment: Puysegur)            45-6=39 -> 40 km  (matches official 40)
  kurilsjapan    (dominant segment: Kurile-Kamchatka)  60-8=52 -> 55 km  (matches official 55)
"Dominant segment" = the longer of the two segments PTHA18 merges into one
zone (Tonga is longer than Kermadec; Kurile-Kamchatka is longer than Japan).
This 3/3 exact match is what justifies using max-of-dominant-segment rather
than e.g. max-of-all-segments (which gives 60 km for kurilsjapan, wrong by
one rounding step) or pref-of-dominant-segment (which gives materially
smaller cutoffs, e.g. 30-40 km, not matching any official value).

BERRYMAN_TABLE_SOURCE
----------------------
Numbers below were read from Berryman.pdf (downloaded by the user from the
National Library of New Zealand's digital heritage archive,
ndhadeliver.natlib.govt.nz, item IE26233616 / file FL26233618) via
`pdftotext -layout`, then cross-checked by rendering the actual PDF pages
(pdftoppm, 200 dpi) and reading the table image directly -- required because
pdftotext -layout misaligns columns on several rows of this table (confirmed
by direct comparison; wrong values from pdftotext alone were caught and
fixed against the rendered images while extracting the full table below).
Table 3.1 (geometry + coupling-pref) spans:
  page 25 (rows 1-19, incl. Japan/Kurile)
  page 26 (rows 20-40, incl. Hikurangi-Tonga-Kermadec, Puysegur)
  page 27 (rows 41-59)
  page 28 (rows 60-76)
  page 29 (rows 77-79, end of table)
The coupling/Mmax/b-value table (same row numbering, separate columns) spans:
  page 30 (rows 1-21)
  page 31 (rows 22-45)
  page 32 (rows 46-66)
  page 33 (rows 67-79, end of table)
All 79 rows of both tables were transcribed this way (image reading as the
final authority whenever it disagreed with pdftotext).

Two anomalies are printed in the source PDF itself (not transcription
errors -- verified against the rendered page images both times):
  - Row 47 (Hellenic Tr. Whole Margin) prints trench_depth_km = 24, which is
    almost certainly a typo for 2.4 (row 63, Mexico/CA Whole Margin, has
    identical dip/updip/down-dip/width/coupling values but trench_depth=2.4)
    -- transcribed as printed, flagged with a comment on that row.
  - Rows 6 and 7 (Alaska/Aleutians Kodiak, Prince William Sound) print
    coupling_pref (0.8) lower than coupling_min (0.9) -- also transcribed as
    printed, flagged with comments on those rows.

Known limitation: BERRYMAN_ROWS now has all 79 Table 3.1 segments, and
ZONE_TO_BERRYMAN_SEGMENTS/ZONE_DOMINANT_SEGMENT cover 28 of the 51 PTHA18
unsegmented zones in rptha's sourcezone_parameters.csv. The remaining zones
(outer-rise sources, backthrusts, detachments, small local thrust systems
with no clearly corresponding Table 3.1 segment, and Macquarie's
transform-boundary zones) are intentionally left unmapped rather than
guessed -- see the comment above ZONE_TO_BERRYMAN_SEGMENTS's closing brace
for the full list and reasoning. Adding one of those means confirming it
truly has no Table 3.1 analogue (Berryman's paper is interface-only) rather
than assuming BERRYMAN_ROWS is missing a row for it.

Zones PTHA18 never modelled at all (new use, v6)
--------------------------------------------------
BERRYMAN_ROWS covers every segment in Table 3.1, including several with no
PTHA18 zone at all (PTHA18 only modelled the Pacific/Indian Ocean basins
threatening Australia; Table 3.1 is global). generate.py --ptha false lets
such a zone run the full pipeline (see its module docstring's "Zones
PTHA18 never modelled" section) provided it gets an entry in
ZONE_TO_BERRYMAN_SEGMENTS/ZONE_DOMINANT_SEGMENT here, keyed by whatever
zone name is passed to --zone (not a PTHA18 sourcename -- there isn't one).
"calabria" is the first such entry, mapped to the existing "Calabria" row
(table_no 46). To add another: find its Table 3.1 row(s) the same way
every row above was transcribed (BERRYMAN_TABLE_SOURCE), add them to
BERRYMAN_ROWS if not already present, then add the zone-name mapping here.
sourcezone_parameters.csv (scaling_relation/shear_modulus/unit-source
width) has no possible row for such a zone -- generic defaults apply
automatically instead (see official_geometry_params.scaling_relation_and_
shear_modulus's docstring).
"""

# One row per Berryman et al. (2015) Table 3.1 segment (all 79 rows of the
# table, pages 25-29 for geometry + coupling-pref, pages 30-33 for
# coupling-min/max, Mmax and b-value -- see BERRYMAN_TABLE_SOURCE above).
#
# depth values are km BELOW SEA LEVEL, exactly as printed in Table 3.1 (NOT
# below trench -- that conversion happens in cutoff_km() below).
BERRYMAN_ROWS = {
    "Kermadec": {
        "table_no": 23,
        "dip_deg": 12.0,
        "trench_depth_km": 7.0,
        "down_dip_depth_pref_km": 30.0,
        "down_dip_depth_min_km": 15.0,
        "down_dip_depth_max_km": 40.0,
        "coupling_pref": 0.3,
        "coupling_min": 0.20,
        "coupling_max": 0.75,
        "mw_max_pref": 8.76,
        "mw_max_min": 8.10,
        "mw_max_max": 9.42,
        "b_pref": 0.95,
        "b_min": 0.70,
        "b_max": 1.20,
    },
    "Tonga": {
        "table_no": 24,
        "dip_deg": 17.0,
        "trench_depth_km": 8.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.2,
        "coupling_min": 0.10,
        "coupling_max": 0.70,
        "mw_max_pref": 8.58,
        "mw_max_min": 8.00,
        "mw_max_max": 9.17,
        "b_pref": 0.95,
        "b_min": 0.70,
        "b_max": 1.20,
    },
    "Puysegur": {
        "table_no": 25,
        "dip_deg": 15.0,
        "trench_depth_km": 6.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 30.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.7,
        "coupling_min": 0.50,
        "coupling_max": 0.80,
        "mw_max_pref": 8.43,
        "mw_max_min": 7.80,
        "mw_max_max": 9.07,
        "b_pref": 0.95,
        "b_min": 0.70,
        "b_max": 1.20,
    },
    "Japan": {
        "table_no": 11,
        "dip_deg": 15.0,
        "trench_depth_km": 7.0,
        "down_dip_depth_pref_km": 50.0,
        "down_dip_depth_min_km": 40.0,
        "down_dip_depth_max_km": 65.0,
        "coupling_pref": 0.7,
        "coupling_min": 0.60,
        "coupling_max": 0.90,
        "mw_max_pref": 9.08,
        "mw_max_min": 9.00,
        "mw_max_max": 9.16,
        "b_pref": 0.91,
        "b_min": 0.61,
        "b_max": 1.20,
    },
    "Kurile-Kamchatka": {
        "table_no": 12,
        "dip_deg": 16.0,
        "trench_depth_km": 8.0,
        "down_dip_depth_pref_km": 50.0,
        "down_dip_depth_min_km": 40.0,
        "down_dip_depth_max_km": 60.0,
        "coupling_pref": 0.8,
        "coupling_min": 0.70,
        "coupling_max": 0.90,
        "mw_max_pref": 9.30,
        "mw_max_min": 9.00,
        "mw_max_max": 9.60,
        "b_pref": 0.92,
        "b_min": 0.63,
        "b_max": 1.20,
    },
    "Alaska/Aleutians Whole Margin": {
        "table_no": 1,
        "dip_deg": 14.0,
        "trench_depth_km": 6.0,
        "down_dip_depth_pref_km": 40.0,
        "down_dip_depth_min_km": 26.0,
        "down_dip_depth_max_km": 48.0,
        "coupling_pref": 0.55,
        "coupling_min": 0.42,
        "coupling_max": 0.77,
        "mw_max_pref": 9.4,
        "mw_max_min": 9.2,
        "mw_max_max": 9.6,
        "b_pref": 0.93,
        "b_min": 0.67,
        "b_max": 1.2,
    },
    "Alaska/Aleutians Komandorski": {
        "table_no": 2,
        "dip_deg": 15.0,
        "trench_depth_km": 5.5,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.4,
        "mw_max_min": 8.0,
        "mw_max_max": 8.8,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Alaska/Aleutians Western Aleutians": {
        "table_no": 3,
        "dip_deg": 18.0,
        "trench_depth_km": 7.0,
        "down_dip_depth_pref_km": 50.0,
        "down_dip_depth_min_km": 30.0,
        "down_dip_depth_max_km": 55.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 9.4,
        "mw_max_min": 9.2,
        "mw_max_max": 9.6,
        "b_pref": 0.92,
        "b_min": 0.63,
        "b_max": 1.2,
    },
    "Alaska/Aleutians Shumagin": {
        "table_no": 4,
        "dip_deg": 14.0,
        "trench_depth_km": 6.0,
        "down_dip_depth_pref_km": 26.0,
        "down_dip_depth_min_km": 20.0,
        "down_dip_depth_max_km": 32.0,
        "coupling_pref": 0.2,
        "coupling_min": 0.1,
        "coupling_max": 0.7,
        "mw_max_pref": 7.93,
        "mw_max_min": 7.5,
        "mw_max_max": 8.35,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Alaska/Aleutians Semidi": {
        "table_no": 5,
        "dip_deg": 14.0,
        "trench_depth_km": 6.0,
        "down_dip_depth_pref_km": 30.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 50.0,
        "coupling_pref": 0.7,
        "coupling_min": 0.6,
        "coupling_max": 0.9,
        "mw_max_pref": 8.5,
        "mw_max_min": 8.34,
        "mw_max_max": 8.5,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Alaska/Aleutians Kodiak": {
        "table_no": 6,
        "dip_deg": 8.0,
        "trench_depth_km": 4.5,
        "down_dip_depth_pref_km": 28.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 50.0,
        # coupling_pref (0.8) prints lower than coupling_min (0.9) in the PDF
        # itself (Table 3.1 last column vs the coupling/Mmax/b table) -- an
        # anomaly in the source, not a transcription error; verified against
        # both tables' rendered page images.
        "coupling_pref": 0.8,
        "coupling_min": 0.9,
        "coupling_max": 1.0,
        "mw_max_pref": 9.2,
        "mw_max_min": 8.63,
        "mw_max_max": 9.2,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Alaska/Aleutians Prince William Sound": {
        "table_no": 7,
        "dip_deg": 6.0,
        "trench_depth_km": 4.5,
        "down_dip_depth_pref_km": 42.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 50.0,
        # same coupling_pref < coupling_min anomaly as Kodiak above, as
        # printed in the source PDF.
        "coupling_pref": 0.8,
        "coupling_min": 0.9,
        "coupling_max": 1.0,
        "mw_max_pref": 9.2,
        "mw_max_min": 9.0,
        "mw_max_max": 9.2,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Alaska/Aleutians Yakataga": {
        "table_no": 8,
        "dip_deg": 15.0,
        "trench_depth_km": 4.0,
        "down_dip_depth_pref_km": 15.0,
        "down_dip_depth_min_km": 10.0,
        "down_dip_depth_max_km": 20.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.1,
        "mw_max_min": 8.0,
        "mw_max_max": 8.1,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Cascadia": {
        "table_no": 9,
        "dip_deg": 15.0,
        "trench_depth_km": 2.5,
        "down_dip_depth_pref_km": 25.0,
        "down_dip_depth_min_km": 20.0,
        "down_dip_depth_max_km": 30.0,
        "coupling_pref": 0.8,
        "coupling_min": 0.7,
        "coupling_max": 0.9,
        "mw_max_pref": 9.0,
        "mw_max_min": 8.8,
        "mw_max_max": 9.2,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Japan/Kurile Whole Margin": {
        "table_no": 10,
        "dip_deg": 16.0,
        "trench_depth_km": 8.0,
        "down_dip_depth_pref_km": 50.0,
        "down_dip_depth_min_km": 40.0,
        "down_dip_depth_max_km": 61.0,
        "coupling_pref": 0.77,
        "coupling_min": 0.67,
        "coupling_max": 0.9,
        "mw_max_pref": 9.3,
        "mw_max_min": 9.0,
        "mw_max_max": 9.6,
        "b_pref": 0.91,
        "b_min": 0.62,
        "b_max": 1.2,
    },
    "Kanto": {
        "table_no": 13,
        "dip_deg": 15.0,
        "trench_depth_km": 1.0,
        "down_dip_depth_pref_km": 25.0,
        "down_dip_depth_min_km": 20.0,
        "down_dip_depth_max_km": 30.0,
        "coupling_pref": 0.9,
        "coupling_min": 0.8,
        "coupling_max": 1.0,
        "mw_max_pref": 8.21,
        "mw_max_min": 8.0,
        "mw_max_max": 8.42,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Nankai/Ryukyu Whole Margin": {
        "table_no": 14,
        "dip_deg": 15.0,
        "trench_depth_km": 5.0,
        "down_dip_depth_pref_km": 22.0,
        "down_dip_depth_min_km": 17.0,
        "down_dip_depth_max_km": 27.0,
        "coupling_pref": 0.44,
        "coupling_min": 0.34,
        "coupling_max": 0.8,
        "mw_max_pref": 8.95,
        "mw_max_min": 8.5,
        "mw_max_max": 9.41,
        "b_pref": 0.91,
        "b_min": 0.61,
        "b_max": 1.2,
    },
    "Nankai/Ryukyu Nankai": {
        "table_no": 15,
        "dip_deg": 15.0,
        "trench_depth_km": 3.5,
        "down_dip_depth_pref_km": 25.0,
        "down_dip_depth_min_km": 20.0,
        "down_dip_depth_max_km": 30.0,
        "coupling_pref": 0.9,
        "coupling_min": 0.8,
        "coupling_max": 1.0,
        "mw_max_pref": 8.7,
        "mw_max_min": 8.5,
        "mw_max_max": 8.9,
        "b_pref": 0.91,
        "b_min": 0.61,
        "b_max": 1.2,
    },
    "Nankai/Ryukyu Ryukyu": {
        "table_no": 16,
        "dip_deg": 15.0,
        "trench_depth_km": 6.0,
        "down_dip_depth_pref_km": 20.0,
        "down_dip_depth_min_km": 15.0,
        "down_dip_depth_max_km": 25.0,
        "coupling_pref": 0.2,
        "coupling_min": 0.1,
        "coupling_max": 0.7,
        "mw_max_pref": 8.54,
        "mw_max_min": 8.0,
        "mw_max_max": 9.09,
        "b_pref": 0.91,
        "b_min": 0.61,
        "b_max": 1.2,
    },
    "Izu-Bonin": {
        "table_no": 17,
        "dip_deg": 15.0,
        "trench_depth_km": 7.5,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.2,
        "coupling_min": 0.1,
        "coupling_max": 0.7,
        "mw_max_pref": 8.21,
        "mw_max_min": 7.2,
        "mw_max_max": 9.21,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Marianas": {
        "table_no": 18,
        "dip_deg": 15.0,
        "trench_depth_km": 8.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.2,
        "coupling_min": 0.1,
        "coupling_max": 0.7,
        "mw_max_pref": 8.34,
        "mw_max_min": 7.2,
        "mw_max_max": 9.48,
        "b_pref": 1.08,
        "b_min": 0.68,
        "b_max": 1.47,
    },
    "North Yap": {
        "table_no": 19,
        "dip_deg": 15.0,
        "trench_depth_km": 7.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.2,
        "coupling_min": 0.1,
        "coupling_max": 0.7,
        "mw_max_pref": 8.07,
        "mw_max_min": 7.2,
        "mw_max_max": 8.93,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Palau-South Yap": {
        "table_no": 20,
        "dip_deg": 15.0,
        "trench_depth_km": 5.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.2,
        "coupling_min": 0.1,
        "coupling_max": 0.7,
        "mw_max_pref": 8.02,
        "mw_max_min": 7.2,
        "mw_max_max": 8.83,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Hikurangi-Tonga-Kermadec Whole margin": {
        "table_no": 21,
        "dip_deg": 13.0,
        "trench_depth_km": 6.0,
        "down_dip_depth_pref_km": 32.0,
        "down_dip_depth_min_km": 20.0,
        "down_dip_depth_max_km": 41.0,
        "coupling_pref": 0.31,
        "coupling_min": 0.21,
        "coupling_max": 0.72,
        "mw_max_pref": 8.85,
        "mw_max_min": 8.1,
        "mw_max_max": 9.6,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.21,
    },
    "H-K-T Hikurangi": {
        "table_no": 22,
        "dip_deg": 10.0,
        "trench_depth_km": 2.5,
        "down_dip_depth_pref_km": 30.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 35.0,
        "coupling_pref": 0.54,
        "coupling_min": 0.4,
        "coupling_max": 0.7,
        "mw_max_pref": 8.5,
        "mw_max_min": 8.0,
        "mw_max_max": 9.0,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Hjort": {
        "table_no": 26,
        "dip_deg": 22.0,
        "trench_depth_km": 6.0,
        "down_dip_depth_pref_km": 20.0,
        "down_dip_depth_min_km": 15.0,
        "down_dip_depth_max_km": 25.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 7.78,
        "mw_max_min": 7.2,
        "mw_max_max": 8.36,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Solomons Whole margin": {
        "table_no": 27,
        "dip_deg": 26.0,
        "trench_depth_km": 3.6,
        "down_dip_depth_pref_km": 40.0,
        "down_dip_depth_min_km": 35.0,
        "down_dip_depth_max_km": 60.0,
        "coupling_pref": 0.7,
        "coupling_min": 0.6,
        "coupling_max": 0.8,
        "mw_max_pref": 8.7,
        "mw_max_min": 8.1,
        "mw_max_max": 9.31,
        "b_pref": 0.9,
        "b_min": 0.6,
        "b_max": 1.2,
    },
    "Solomon Northwest": {
        "table_no": 28,
        "dip_deg": 26.0,
        "trench_depth_km": 3.6,
        "down_dip_depth_pref_km": 40.0,
        "down_dip_depth_min_km": 35.0,
        "down_dip_depth_max_km": 60.0,
        "coupling_pref": 0.7,
        "coupling_min": 0.6,
        "coupling_max": 0.8,
        "mw_max_pref": 8.36,
        "mw_max_min": 8.1,
        "mw_max_max": 8.62,
        "b_pref": 0.9,
        "b_min": 0.6,
        "b_max": 1.2,
    },
    "Solomon Southeast": {
        "table_no": 29,
        "dip_deg": 26.0,
        "trench_depth_km": 2.5,
        "down_dip_depth_pref_km": 40.0,
        "down_dip_depth_min_km": 35.0,
        "down_dip_depth_max_km": 60.0,
        "coupling_pref": 0.7,
        "coupling_min": 0.6,
        "coupling_max": 0.8,
        "mw_max_pref": 8.6,
        "mw_max_min": 8.1,
        "mw_max_max": 9.09,
        "b_pref": 0.9,
        "b_min": 0.6,
        "b_max": 1.2,
    },
    "New Hebrides Whole Margin": {
        "table_no": 30,
        "dip_deg": 24.0,
        "trench_depth_km": 6.0,
        "down_dip_depth_pref_km": 31.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 35.0,
        "coupling_pref": 0.37,
        "coupling_min": 0.27,
        "coupling_max": 0.73,
        "mw_max_pref": 8.83,
        "mw_max_min": 8.3,
        "mw_max_max": 9.37,
        "b_pref": 0.9,
        "b_min": 0.6,
        "b_max": 1.2,
    },
    "New Hebrides North": {
        "table_no": 31,
        "dip_deg": 23.0,
        "trench_depth_km": 6.0,
        "down_dip_depth_pref_km": 30.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 40.0,
        "coupling_pref": 0.25,
        "coupling_min": 0.15,
        "coupling_max": 0.7,
        "mw_max_pref": 8.02,
        "mw_max_min": 7.6,
        "mw_max_max": 8.44,
        "b_pref": 0.9,
        "b_min": 0.6,
        "b_max": 1.2,
    },
    "New Hebrides Central": {
        "table_no": 32,
        "dip_deg": 23.0,
        "trench_depth_km": 6.0,
        "down_dip_depth_pref_km": 40.0,
        "down_dip_depth_min_km": 30.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.7,
        "coupling_min": 0.6,
        "coupling_max": 0.8,
        "mw_max_pref": 8.5,
        "mw_max_min": 8.3,
        "mw_max_max": 8.7,
        "b_pref": 0.9,
        "b_min": 0.6,
        "b_max": 1.2,
    },
    "New Hebrides South": {
        "table_no": 33,
        "dip_deg": 23.0,
        "trench_depth_km": 6.0,
        "down_dip_depth_pref_km": 30.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 40.0,
        "coupling_pref": 0.25,
        "coupling_min": 0.15,
        "coupling_max": 0.7,
        "mw_max_pref": 8.12,
        "mw_max_min": 7.6,
        "mw_max_max": 8.64,
        "b_pref": 0.9,
        "b_min": 0.6,
        "b_max": 1.2,
    },
    "New Hebrides Matthew-Hunter": {
        "table_no": 34,
        "dip_deg": 28.0,
        "trench_depth_km": 6.0,
        "down_dip_depth_pref_km": 25.0,
        "down_dip_depth_min_km": 20.0,
        "down_dip_depth_max_km": 30.0,
        "coupling_pref": 0.25,
        "coupling_min": 0.15,
        "coupling_max": 0.7,
        "mw_max_pref": 8.19,
        "mw_max_min": 8.0,
        "mw_max_max": 8.39,
        "b_pref": 0.9,
        "b_min": 0.6,
        "b_max": 1.2,
    },
    "New Britain": {
        "table_no": 35,
        "dip_deg": 26.0,
        "trench_depth_km": 6.5,
        "down_dip_depth_pref_km": 40.0,
        "down_dip_depth_min_km": 30.0,
        "down_dip_depth_max_km": 50.0,
        "coupling_pref": 0.7,
        "coupling_min": 0.6,
        "coupling_max": 0.8,
        "mw_max_pref": 8.41,
        "mw_max_min": 8.0,
        "mw_max_max": 8.82,
        "b_pref": 0.9,
        "b_min": 0.6,
        "b_max": 1.2,
    },
    "New Guinea Trench Whole Margin": {
        "table_no": 36,
        "dip_deg": 15.0,
        "trench_depth_km": 3.6,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.7,
        "coupling_min": 0.6,
        "coupling_max": 0.8,
        "mw_max_pref": 8.78,
        "mw_max_min": 8.2,
        "mw_max_max": 9.37,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "New Guinea Trench East": {
        "table_no": 37,
        "dip_deg": 15.0,
        "trench_depth_km": 3.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.7,
        "coupling_min": 0.6,
        "coupling_max": 0.8,
        "mw_max_pref": 8.25,
        "mw_max_min": 7.6,
        "mw_max_max": 8.9,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "New Guinea Trench West": {
        "table_no": 38,
        "dip_deg": 15.0,
        "trench_depth_km": 4.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.7,
        "coupling_min": 0.6,
        "coupling_max": 0.8,
        "mw_max_pref": 8.61,
        "mw_max_min": 8.2,
        "mw_max_max": 9.03,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Manus Whole Margin": {
        "table_no": 39,
        "dip_deg": 15.0,
        "trench_depth_km": 3.5,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.5,
        "mw_max_min": 7.5,
        "mw_max_max": 9.5,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Manus East": {
        "table_no": 40,
        "dip_deg": 15.0,
        "trench_depth_km": 3.5,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.28,
        "mw_max_min": 7.5,
        "mw_max_max": 9.07,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Manus West": {
        "table_no": 41,
        "dip_deg": 15.0,
        "trench_depth_km": 4.4,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.31,
        "mw_max_min": 7.5,
        "mw_max_max": 9.13,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Andaman-Sunda Trench Whole Margin": {
        "table_no": 42,
        "dip_deg": 14.0,
        "trench_depth_km": 4.4,
        "down_dip_depth_pref_km": 32.0,
        "down_dip_depth_min_km": 26.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.54,
        "coupling_min": 0.44,
        "coupling_max": 0.79,
        "mw_max_pref": 9.3,
        "mw_max_min": 9.0,
        "mw_max_max": 9.6,
        "b_pref": 0.94,
        "b_min": 0.67,
        "b_max": 1.2,
    },
    "Andaman-Sunda Andaman": {
        "table_no": 43,
        "dip_deg": 14.0,
        "trench_depth_km": 3.0,
        "down_dip_depth_pref_km": 40.0,
        "down_dip_depth_min_km": 35.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.7,
        "coupling_min": 0.6,
        "coupling_max": 0.8,
        "mw_max_pref": 9.3,
        "mw_max_min": 9.0,
        "mw_max_max": 9.55,
        "b_pref": 0.94,
        "b_min": 0.67,
        "b_max": 1.2,
    },
    "Andaman-Sunda Sumatra": {
        "table_no": 44,
        "dip_deg": 14.0,
        "trench_depth_km": 5.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 50.0,
        "coupling_pref": 0.8,
        "coupling_min": 0.7,
        "coupling_max": 0.9,
        "mw_max_pref": 9.2,
        "mw_max_min": 9.0,
        "mw_max_max": 9.4,
        "b_pref": 0.94,
        "b_min": 0.67,
        "b_max": 1.2,
    },
    "Andaman-Sunda Java": {
        "table_no": 45,
        "dip_deg": 15.0,
        "trench_depth_km": 5.0,
        "down_dip_depth_pref_km": 25.0,
        "down_dip_depth_min_km": 20.0,
        "down_dip_depth_max_km": 40.0,
        "coupling_pref": 0.2,
        "coupling_min": 0.1,
        "coupling_max": 0.7,
        "mw_max_pref": 8.61,
        "mw_max_min": 7.8,
        "mw_max_max": 9.42,
        "b_pref": 0.94,
        "b_min": 0.67,
        "b_max": 1.2,
    },
    "Calabria": {
        "table_no": 46,
        "dip_deg": 20.0,
        "trench_depth_km": 4.0,
        "down_dip_depth_pref_km": 45.0,
        "down_dip_depth_min_km": 40.0,
        "down_dip_depth_max_km": 60.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 7.74,
        "mw_max_min": 7.1,
        "mw_max_max": 8.38,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Hellenic Tr. Whole Margin": {
        "table_no": 47,
        "dip_deg": 35.0,
        "trench_depth_km": 24.0,  # printed as 24 in the PDF; likely a typo for 2.4 -- row 63 (Mexico/CA Whole Margin) has identical dip/updip/down-dip/width/coupling but trench_depth=2.4. Transcribed as printed, not corrected.
        "down_dip_depth_pref_km": 45.0,
        "down_dip_depth_min_km": 40.0,
        "down_dip_depth_max_km": 50.0,
        "coupling_pref": 0.6,
        "coupling_min": 0.2,
        "coupling_max": 1.0,
        "mw_max_pref": 8.5,
        "mw_max_min": 8.0,
        "mw_max_max": 9.0,
        "b_pref": 0.95,
        "b_min": 0.69,
        "b_max": 1.2,
    },
    "Hellenic western segment": {
        "table_no": 48,
        "dip_deg": 30.0,
        "trench_depth_km": 2.0,
        "down_dip_depth_pref_km": 45.0,
        "down_dip_depth_min_km": 40.0,
        "down_dip_depth_max_km": 50.0,
        "coupling_pref": 0.6,
        "coupling_min": 0.2,
        "coupling_max": 1.0,
        "mw_max_pref": 8.37,
        "mw_max_min": 8.0,
        "mw_max_max": 8.74,
        "b_pref": 0.95,
        "b_min": 0.69,
        "b_max": 1.2,
    },
    "Hellenic eastern segment": {
        "table_no": 49,
        "dip_deg": 42.0,
        "trench_depth_km": 3.0,
        "down_dip_depth_pref_km": 45.0,
        "down_dip_depth_min_km": 40.0,
        "down_dip_depth_max_km": 50.0,
        "coupling_pref": 0.6,
        "coupling_min": 0.2,
        "coupling_max": 1.0,
        "mw_max_pref": 8.21,
        "mw_max_min": 8.0,
        "mw_max_max": 8.42,
        "b_pref": 0.95,
        "b_min": 0.69,
        "b_max": 1.2,
    },
    "Cyprus western segment": {
        "table_no": 50,
        "dip_deg": 39.0,
        "trench_depth_km": 2.0,
        "down_dip_depth_pref_km": 50.0,
        "down_dip_depth_min_km": 40.0,
        "down_dip_depth_max_km": 60.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.0,
        "mw_max_min": 7.5,
        "mw_max_max": 8.49,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Cyprus eastern segment": {
        "table_no": 51,
        "dip_deg": 42.0,
        "trench_depth_km": 2.0,
        "down_dip_depth_pref_km": 50.0,
        "down_dip_depth_min_km": 40.0,
        "down_dip_depth_max_km": 60.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 7.89,
        "mw_max_min": 7.5,
        "mw_max_max": 8.29,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Makran": {
        "table_no": 52,
        "dip_deg": 8.0,
        "trench_depth_km": 3.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 30.0,
        "down_dip_depth_max_km": 40.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.71,
        "mw_max_min": 8.1,
        "mw_max_max": 9.33,
        "b_pref": 0.95,
        "b_min": 0.69,
        "b_max": 1.2,
    },
    "S. America Whole Margin": {
        "table_no": 53,
        "dip_deg": 14.0,
        "trench_depth_km": 5.0,
        "down_dip_depth_pref_km": 50.0,
        "down_dip_depth_min_km": 40.0,
        "down_dip_depth_max_km": 60.0,
        "coupling_pref": 0.8,
        "coupling_min": 0.7,
        "coupling_max": 0.9,
        "mw_max_pref": 9.55,
        "mw_max_min": 9.5,
        "mw_max_max": 9.6,
        "b_pref": 0.88,
        "b_min": 0.56,
        "b_max": 1.2,
    },
    "S. America Ecuador-Colombia": {
        "table_no": 54,
        "dip_deg": 15.0,
        "trench_depth_km": 3.0,
        "down_dip_depth_pref_km": 50.0,
        "down_dip_depth_min_km": 40.0,
        "down_dip_depth_max_km": 60.0,
        "coupling_pref": 0.8,
        "coupling_min": 0.7,
        "coupling_max": 0.9,
        "mw_max_pref": 9.14,
        "mw_max_min": 8.8,
        "mw_max_max": 9.49,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "S. America Peru": {
        "table_no": 55,
        "dip_deg": 13.0,
        "trench_depth_km": 5.0,
        "down_dip_depth_pref_km": 50.0,
        "down_dip_depth_min_km": 40.0,
        "down_dip_depth_max_km": 60.0,
        "coupling_pref": 0.8,
        "coupling_min": 0.7,
        "coupling_max": 0.9,
        "mw_max_pref": 9.3,
        "mw_max_min": 9.0,
        "mw_max_max": 9.6,
        "b_pref": 0.87,
        "b_min": 0.53,
        "b_max": 1.2,
    },
    "S. America N. Chile": {
        "table_no": 56,
        "dip_deg": 15.0,
        "trench_depth_km": 6.0,
        "down_dip_depth_pref_km": 50.0,
        "down_dip_depth_min_km": 40.0,
        "down_dip_depth_max_km": 60.0,
        "coupling_pref": 0.8,
        "coupling_min": 0.7,
        "coupling_max": 0.9,
        "mw_max_pref": 9.05,
        "mw_max_min": 8.6,
        "mw_max_max": 9.49,
        "b_pref": 0.87,
        "b_min": 0.53,
        "b_max": 1.2,
    },
    "S. America Central Chile": {
        "table_no": 57,
        "dip_deg": 12.0,
        "trench_depth_km": 5.0,
        "down_dip_depth_pref_km": 50.0,
        "down_dip_depth_min_km": 40.0,
        "down_dip_depth_max_km": 60.0,
        "coupling_pref": 0.8,
        "coupling_min": 0.7,
        "coupling_max": 0.9,
        "mw_max_pref": 9.51,
        "mw_max_min": 9.5,
        "mw_max_max": 9.53,
        "b_pref": 0.87,
        "b_min": 0.53,
        "b_max": 1.2,
    },
    "Patagonia Whole Margin": {
        "table_no": 58,
        "dip_deg": 15.0,
        "trench_depth_km": 4.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.8,
        "mw_max_min": 8.0,
        "mw_max_max": 9.6,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Patagonia North": {
        "table_no": 59,
        "dip_deg": 15.0,
        "trench_depth_km": 4.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.5,
        "mw_max_min": 8.0,
        "mw_max_max": 9.0,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Patagonia South": {
        "table_no": 60,
        "dip_deg": 15.0,
        "trench_depth_km": 4.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.72,
        "mw_max_min": 8.0,
        "mw_max_max": 9.45,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "South Shetland": {
        "table_no": 61,
        "dip_deg": 15.0,
        "trench_depth_km": 3.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.11,
        "mw_max_min": 7.5,
        "mw_max_max": 8.71,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "South Sandwich": {
        "table_no": 62,
        "dip_deg": 15.0,
        "trench_depth_km": 7.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.2,
        "coupling_min": 0.1,
        "coupling_max": 0.7,
        "mw_max_pref": 8.26,
        "mw_max_min": 7.5,
        "mw_max_max": 9.01,
        "b_pref": 1.09,
        "b_min": 0.7,
        "b_max": 1.48,
    },
    "Mexico/Central America Whole Margin": {
        "table_no": 63,
        "dip_deg": 35.0,
        "trench_depth_km": 2.4,
        "down_dip_depth_pref_km": 45.0,
        "down_dip_depth_min_km": 40.0,
        "down_dip_depth_max_km": 50.0,
        "coupling_pref": 0.6,
        "coupling_min": 0.37,
        "coupling_max": 0.81,
        "mw_max_pref": 8.9,
        "mw_max_min": 8.2,
        "mw_max_max": 9.6,
        "b_pref": 0.91,
        "b_min": 0.62,
        "b_max": 1.2,
    },
    "Mexico/CA Jalisco": {
        "table_no": 64,
        "dip_deg": 16.0,
        "trench_depth_km": 4.0,
        "down_dip_depth_pref_km": 25.0,
        "down_dip_depth_min_km": 20.0,
        "down_dip_depth_max_km": 35.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.34,
        "mw_max_min": 8.2,
        "mw_max_max": 8.49,
        "b_pref": 0.89,
        "b_min": 0.58,
        "b_max": 1.2,
    },
    "Mexico/CA Michoacan-Guatemala": {
        "table_no": 65,
        "dip_deg": 16.0,
        "trench_depth_km": 4.0,
        "down_dip_depth_pref_km": 20.0,
        "down_dip_depth_min_km": 15.0,
        "down_dip_depth_max_km": 30.0,
        "coupling_pref": 0.7,
        "coupling_min": 0.5,
        "coupling_max": 0.9,
        "mw_max_pref": 8.61,
        "mw_max_min": 8.0,
        "mw_max_max": 9.23,
        "b_pref": 0.89,
        "b_min": 0.58,
        "b_max": 1.2,
    },
    "Mexico/CA El Salvador-Nicaragua": {
        "table_no": 66,
        "dip_deg": 21.0,
        "trench_depth_km": 3.5,
        "down_dip_depth_pref_km": 25.0,
        "down_dip_depth_min_km": 20.0,
        "down_dip_depth_max_km": 35.0,
        "coupling_pref": 0.3,
        "coupling_min": 0.1,
        "coupling_max": 0.7,
        "mw_max_pref": 8.29,
        "mw_max_min": 8.0,
        "mw_max_max": 8.58,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Mexico/CA Costa Rica-west Panama": {
        "table_no": 67,
        "dip_deg": 15.0,
        "trench_depth_km": 2.5,
        "down_dip_depth_pref_km": 25.0,
        "down_dip_depth_min_km": 20.0,
        "down_dip_depth_max_km": 35.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.2,
        "mw_max_min": 7.7,
        "mw_max_max": 8.71,
        "b_pref": 0.95,
        "b_min": 0.69,
        "b_max": 1.2,
    },
    "Antilles": {
        "table_no": 68,
        "dip_deg": 15.0,
        "trench_depth_km": 4.5,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.69,
        "mw_max_min": 8.0,
        "mw_max_max": 9.37,
        "b_pref": 0.92,
        "b_min": 0.64,
        "b_max": 1.2,
    },
    "Manila": {
        "table_no": 69,
        "dip_deg": 15.0,
        "trench_depth_km": 4.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.15,
        "coupling_min": 0.05,
        "coupling_max": 0.7,
        "mw_max_pref": 8.25,
        "mw_max_min": 7.6,
        "mw_max_max": 8.9,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Philippine": {
        "table_no": 70,
        "dip_deg": 25.0,
        "trench_depth_km": 6.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.25,
        "coupling_min": 0.1,
        "coupling_max": 0.75,
        "mw_max_pref": 8.45,
        "mw_max_min": 7.6,
        "mw_max_max": 9.3,
        "b_pref": 0.94,
        "b_min": 0.68,
        "b_max": 1.2,
    },
    "East Luzon Trough": {
        "table_no": 71,
        "dip_deg": 20.0,
        "trench_depth_km": 5.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 7.84,
        "mw_max_min": 7.3,
        "mw_max_max": 8.38,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Cotabato Trench": {
        "table_no": 72,
        "dip_deg": 15.0,
        "trench_depth_km": 4.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.19,
        "mw_max_min": 8.0,
        "mw_max_max": 8.38,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Sulu Trench": {
        "table_no": 73,
        "dip_deg": 15.0,
        "trench_depth_km": 4.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.36,
        "mw_max_min": 8.0,
        "mw_max_max": 8.72,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Minahassa Trench": {
        "table_no": 74,
        "dip_deg": 15.0,
        "trench_depth_km": 3.5,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.39,
        "mw_max_min": 7.9,
        "mw_max_max": 8.89,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Seram Trough": {
        "table_no": 75,
        "dip_deg": 15.0,
        "trench_depth_km": 6.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.5,
        "mw_max_min": 8.0,
        "mw_max_max": 9.04,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Timor": {
        "table_no": 76,
        "dip_deg": 15.0,
        "trench_depth_km": 3.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.0,
        "mw_max_min": 7.5,
        "mw_max_max": 9.38,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Manokwari Trench": {
        "table_no": 77,
        "dip_deg": 15.0,
        "trench_depth_km": 4.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.12,
        "mw_max_min": 7.6,
        "mw_max_max": 8.64,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Halmahera": {
        "table_no": 78,
        "dip_deg": 15.0,
        "trench_depth_km": 2.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.33,
        "mw_max_min": 8.3,
        "mw_max_max": 8.35,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
    "Kepulauan Sangihe": {
        "table_no": 79,
        "dip_deg": 15.0,
        "trench_depth_km": 2.0,
        "down_dip_depth_pref_km": 35.0,
        "down_dip_depth_min_km": 25.0,
        "down_dip_depth_max_km": 45.0,
        "coupling_pref": 0.5,
        "coupling_min": 0.3,
        "coupling_max": 0.7,
        "mw_max_pref": 8.39,
        "mw_max_min": 8.3,
        "mw_max_max": 8.47,
        "b_pref": 0.95,
        "b_min": 0.7,
        "b_max": 1.2,
    },
}

# PTHA18 zone -> Berryman segments it merges, dominant segment LAST.
# "Dominant" = the longer along-strike segment, which cutoff_km() below
# uses for the seismogenic-cutoff derivation (see module docstring for the
# 3/3 verification of that rule). coupling/mw_max/b use the SAME weighting
# PTHA18's own sourcezone_parameters.csv unsegmented row implies: an average
# across the merged segments, area-weighted by down-dip width times
# along-strike length is not recoverable from this table alone, so this
# uses a simple mean across segments as the "declared" prior -- same spirit
# as PTHA18's own unsegmented-row treatment (Table 2, Appendix A) which is
# itself described as "50% unsegmented, 50% average of segments".
ZONE_TO_BERRYMAN_SEGMENTS = {
    "kermadectonga2": ["Kermadec", "Tonga"],
    "puysegur2": ["Puysegur"],
    "kurilsjapan": ["Japan", "Kurile-Kamchatka"],
    # --- added when extending BERRYMAN_ROWS to the full 79-row Table 3.1 ---
    # "kermadectonga" and "puysegur" are the unsegmented siblings of
    # "kermadectonga2"/"puysegur2" above (same PTHA18 fault geometry, just
    # not the "2" revision) -- same Berryman segments apply.
    "kermadectonga": ["Kermadec", "Tonga"],
    "puysegur": ["Puysegur"],
    # izumariana: PTHA18's sourcezone_parameters.csv segments this zone into
    # "_izubonin" (along-strike index span 1-27, 27 units) and "_marianas"
    # (28-70, 43 units) -- directly matching Berryman's "Izu-Bonin" and
    # "Marianas" rows. Marianas is longer (43 > 27) -> dominant.
    "izumariana": ["Izu-Bonin", "Marianas"],
    # newhebrides/newhebrides2: PTHA18 segments are "_matthewhunter",
    # "_south", "_central", "_north" -- matching Berryman's 4 New Hebrides
    # segments by name. Along-strike spans (newhebrides): matthewhunter=5,
    # south=11, central=10, north=8 units -> South is dominant.
    # (newhebrides2's spans: 9, 11, 9, 9 -> South is dominant again.)
    "newhebrides": [
        "New Hebrides Matthew-Hunter", "New Hebrides Central",
        "New Hebrides North", "New Hebrides South",
    ],
    "newhebrides2": [
        "New Hebrides Matthew-Hunter", "New Hebrides Central",
        "New Hebrides North", "New Hebrides South",
    ],
    # solomon/solomon2: PTHA18 segments this zone into "_southeast",
    # "_northwest" and "_newbritain" -- the first two match Berryman's
    # Solomon Northwest/Southeast rows directly; "_newbritain" is matched to
    # Berryman's separate "New Britain" row (table_no 35, not filed under
    # "Solomons" in Table 3.1, but PTHA18 merges it into this zone). Spans
    # (solomon): southeast=20, northwest=10, newbritain=16 -> Southeast
    # dominant. (solomon2: 19, 10, 13 -> Southeast dominant again.)
    "solomon": ["Solomon Northwest", "New Britain", "Solomon Southeast"],
    "solomon2": ["Solomon Northwest", "New Britain", "Solomon Southeast"],
    # southamerica: PTHA18 segments are "_patagonia", "_central_chile",
    # "_northern_chile", "_peru", "_ecuador". Berryman's "S. America" table
    # (rows 53-57) only covers Ecuador-Colombia/Peru/N.Chile/Central Chile --
    # "_patagonia" corresponds to Berryman's separate "Patagonia" zone
    # (rows 58-60, a different Table 3.1 grouping), so it is NOT included
    # here (would mix two different Berryman zone-groupings into one
    # PTHA18 zone's average). Spans of the 4 included segments: ecuador=28,
    # peru=49, northern_chile=28, central_chile=26 -> Peru dominant.
    "southamerica": [
        "S. America Ecuador-Colombia", "S. America N. Chile",
        "S. America Central Chile", "S. America Peru",
    ],
    # sunda/sunda2: PTHA18 segments are "_java", "_sumatra", "_andaman",
    # "_arakan". Berryman's Andaman-Sunda table only has Andaman/Sumatra/Java
    # rows (43-45) -- no "Arakan" row exists in Table 3.1, so "_arakan" is
    # left out of the average. Spans (sunda): java=39, sumatra=29,
    # andaman=29 -> Java dominant. (sunda2: 39, 28, 29 -> Java dominant.)
    "sunda": ["Andaman-Sunda Andaman", "Andaman-Sunda Sumatra", "Andaman-Sunda Java"],
    "sunda2": ["Andaman-Sunda Andaman", "Andaman-Sunda Sumatra", "Andaman-Sunda Java"],
    # cascadia, ryuku, hjort, makran, makran2, timor, timortrough, sangihe,
    # manokwari, philippine: PTHA18 does not segment these zones further
    # (sourcezone_parameters.csv has only the unsegmented row), so each maps
    # to its single matching Berryman row.
    "cascadia": ["Cascadia"],
    "hjort": ["Hjort"],
    "makran": ["Makran"],
    "makran2": ["Makran"],
    "timor": ["Timor"],
    "timortrough": ["Timor"],
    "sangihe": ["Kepulauan Sangihe"],
    "manokwari": ["Manokwari Trench"],
    "philippine": ["Philippine"],
    # ryuku: PTHA18 segments this zone into "_nankai" (1-13, 13 units) and
    # "_ryuku" (14-40, 27 units), matching Berryman's Nankai/Ryukyu rows
    # directly. Ryukyu is longer -> dominant.
    "ryuku": ["Nankai/Ryukyu Nankai", "Nankai/Ryukyu Ryukyu"],
    # mexico, newguinea, newguinea2, manus: PTHA18 has no internal
    # segmentation for these zones (only the unsegmented row), even though
    # Berryman's table further subdivides the corresponding margin. Rather
    # than pick one Berryman sub-segment as "dominant" for a zone PTHA18
    # itself treats as a single fault, this uses Berryman's own
    # "Whole Margin" row for that margin, which is the one Berryman row
    # actually representing the zone's full extent.
    "mexico": ["Mexico/Central America Whole Margin"],
    "sandwich": ["South Sandwich"],
    "newguinea": ["New Guinea Trench Whole Margin"],
    "newguinea2": ["New Guinea Trench Whole Margin"],
    "manus": ["Manus Whole Margin"],
    # alaskaaleutians: PTHA18 segments this zone into only 3 pieces
    # ("_eastern", "_western", "_komandorski") that do not line up 1:1 with
    # Berryman's 7 named segments (Komandorski, Western Aleutians, Shumagin,
    # Semidi, Kodiak, Prince William Sound, Yakataga) -- e.g. PTHA18's
    # "_western" (30-67) likely spans several of Berryman's named segments
    # at once. Rather than guess a segment-by-segment correspondence, this
    # uses Berryman's own "Whole Margin" row, the one unambiguous match.
    "alaskaaleutians": ["Alaska/Aleutians Whole Margin"],
    # --- zones PTHA18 never modelled (no sourcezone_parameters.csv row at
    # all), added so generate.py --ptha false can still run the full
    # pipeline for them using this module's own literature data. Unlike
    # every zone above, these have no PTHA18 unsegmented-row name to key
    # off of -- the key here is simply the name passed to generate.py
    # --zone, chosen to match the Berryman row it corresponds to.
    "calabria": ["Calabria"],
    # "calabria2": Calabria's SLAB depth grid turned out to be published
    # under a "Slab2_cal" ScienceBase item (cal_slab2_dep_*.grd), not
    # SLAB1.0 as the "no trailing 2 -> SLAB1.0" naming guess assumed -- so
    # the zone name actually used ends in "2" to route it through SLAB2.0
    # (see slab_product_for_zone). Same Berryman row as "calabria".
    "calabria2": ["Calabria"],
    # "antilles2": same situation as calabria2 -- the Caribbean/Antilles
    # interface's SLAB depth grid is published under the "Caribbean"
    # ScienceBase item (car_slab2_dep_*.grd), SLAB2.0 only; no
    # car_slab1.0_clip.grd exists on USGS's SLAB1.0 file store (confirmed:
    # HTTP 403/AccessDenied, not a Cloudflare bot-check page -- the object
    # genuinely is not there, consistent with Antilles/Caribbean not
    # appearing anywhere in ReportPTHA.pdf, since PTHA18 never modelled
    # this zone at all). So the zone name ends in "2" to route it through
    # SLAB2.0 (see slab_product_for_zone).
    "antilles2": ["Antilles"],
    # Hellenic arc run as two separate zones, each with its own Berryman row.
    "hellenic2": ["Hellenic western segment", "Hellenic eastern segment"],
    "hellenic_west2": ["Hellenic western segment"],
    "hellenic_east2": ["Hellenic eastern segment"],
}

# Dominant (longer) segment per zone, used only for the cutoff derivation.
ZONE_DOMINANT_SEGMENT = {
    "kermadectonga2": "Tonga",
    "puysegur2": "Puysegur",
    "kurilsjapan": "Kurile-Kamchatka",
    "kermadectonga": "Tonga",
    "puysegur": "Puysegur",
    "izumariana": "Marianas",
    "newhebrides": "New Hebrides South",
    "newhebrides2": "New Hebrides South",
    "solomon": "Solomon Southeast",
    "solomon2": "Solomon Southeast",
    "southamerica": "S. America Peru",
    "sunda": "Andaman-Sunda Java",
    "sunda2": "Andaman-Sunda Java",
    "cascadia": "Cascadia",
    "hjort": "Hjort",
    "makran": "Makran",
    "makran2": "Makran",
    "timor": "Timor",
    "timortrough": "Timor",
    "sangihe": "Kepulauan Sangihe",
    "manokwari": "Manokwari Trench",
    "philippine": "Philippine",
    "ryuku": "Nankai/Ryukyu Ryukyu",
    "mexico": "Mexico/Central America Whole Margin",
    "sandwich": "South Sandwich",
    "newguinea": "New Guinea Trench Whole Margin",
    "newguinea2": "New Guinea Trench Whole Margin",
    "manus": "Manus Whole Margin",
    "alaskaaleutians": "Alaska/Aleutians Whole Margin",
    # --- zones PTHA18 never modelled, see ZONE_TO_BERRYMAN_SEGMENTS above ---
    "calabria": "Calabria",
    "calabria2": "Calabria",
    "antilles2": "Antilles",
    "hellenic2": "Hellenic western segment",
    "hellenic_west2": "Hellenic western segment",
    "hellenic_east2": "Hellenic eastern segment",
}

# Zones from the 51-zone PTHA18 unsegmented-row list this module does NOT
# map, and why (see also _segments()'s error message, which lists this):
#   - Outer-rise zones (outerrisesunda, outerrisesolomon,
#     outerrisenewhebrides, outerrise_kermadectonga, outer_rise_timor,
#     outer_rise_puysegur, outerrisesolomon): outer-rise (normal-fault,
#     upper-plate-bending) sources, not subduction interface segments --
#     Berryman's Table 3.1 is interface-only, no matching row exists.
#   - Backthrusts/detachments (sangihe_backthrust, banda_detachment,
#     tolo_thrust): not subduction interface geometry either.
#   - macquarieislandnorth, macquarienorth: Macquarie is a transform/strike
#     -slip boundary, not a subduction interface -- outside Berryman's scope
#     entirely.
#   - arutrough, flores, floreswetar, seram_thrust, se_sulawesi, tanimbar,
#     seramsouth, north_sulawesi: minor/local eastern-Indonesia thrust
#     systems with no dedicated row or clearly corresponding named segment
#     in Table 3.1 (Berryman's Banda-arc coverage is limited to Timor,
#     Seram Trough, Halmahera and Kepulauan Sangihe, already mapped above).
#   - mussau, moresby_trough, trobriand: small Bismarck/Solomon Sea features
#     not named anywhere in Table 3.1.
# These are left unmapped rather than guessed; _segments()/berryman_cutoff_km
# etc. will raise SystemExit with a clear message if called for them.


# Fallbacks for a zone PTHA18 never modelled and that has no Berryman et al.
# (2015) segment mapping in this module. NOT invented: [0.1, 1.3] is the
# SAME uniform coupling prior config.R applies to every zone as half of its
# 50/50 blend regardless of that zone's own spreadsheet value (see
# pyptha_v12/moment_balance.py, step6_write_input.py.tmpl's "uniform_range");
# [0.7, 0.95, 1.2] is the b-value range most Berryman rows already carry
# when Table 3.1 gives "no clear literature differentiation" for a segment
# (e.g. Cascadia, Makran, Timor, Manus). Both are genuine PTHA18/Berryman
# generic defaults already in use elsewhere in this codebase for mapped
# zones, not new numbers invented for this fallback.
# v9 (audit 2026-09-30): the coupling of a zone Berryman et al. (2015) do not
# treat is THEIR OWN default range, [0.3-0.7] ("Where little information is
# available Berryman et al. (2015) propose a default coupling range of
# [0.3-0.7]. In the PTHA18, the latter value is also applied to source-zones
# that are not treated explicitly", ReportPTHA Section 3.7.2.3), with 0.5 as
# the preferred value, as in every such row of PTHA18's table (arutrough,
# flores, north_sulawesi, mussau, trobriand, ...: 0.3/0.5/0.7). Before this it
# was [0.1, 1.3], the uniform half of the prior, which made the 50/50 prior
# entirely uniform.
GENERIC_COUPLING_RANGE = [0.3, 0.5, 0.7]
GENERIC_B_ANCHOR = [0.7, 0.95, 1.2]


def _segments(zone):
    return [BERRYMAN_ROWS[name] for name in ZONE_TO_BERRYMAN_SEGMENTS.get(zone, [])]


def berryman_cutoff_km(zone):
    """Seismogenic depth cutoff below the TRENCH, PTHA18 convention,
    derived the way ReportPTHA.pdf says PTHA18 itself derived it: dominant
    segment's down-dip-depth MAX minus its trench depth, rounded up to a
    multiple of 5 km. See module docstring for the 3/3 exact-match
    verification against official cutoffs.

    If `zone` has no dominant-segment mapping (a zone PTHA18 never modelled
    and this module has no Berryman literature row for), there is no
    generic default for this -- it is genuinely zone-specific geometry.
    Returns (None, None); the caller must supply seismogenic_cutoff_km
    itself (e.g. by hand-editing the generated input JSON) before step 7
    can produce real numbers.
    """
    import math

    dominant = ZONE_DOMINANT_SEGMENT.get(zone)
    if dominant is None:
        return None, None
    row = BERRYMAN_ROWS[dominant]
    below_trench = row["down_dip_depth_max_km"] - row["trench_depth_km"]
    return math.ceil(below_trench / 5.0) * 5.0, dominant


# v9 (2026-09-30): Berryman et al. (2015)'s own row for the WHOLE margin of a
# zone that Table 3.1 also splits into segments. Berryman gives these values
# for the zone as a whole, so v9 uses them for the zone's coupling instead of
# the mean of the segment rows (which is what v8 does). PTHA18 does the same:
# its sourcezone_parameters.csv coupling equals the Whole Margin row on all 11
# of its zones that have one (kermadectonga2 0.21/0.31/0.72, kurilsjapan
# 0.67/0.77/0.9, ryuku 0.34/0.44/0.8, sunda2 0.44/0.54/0.79, newhebrides2
# 0.27/0.37/0.73, ...). Zones already mapped to a Whole Margin row alone
# (alaskaaleutians, mexico, newguinea2, manus) need no entry; izumariana has
# no Whole Margin row, so it keeps the mean of Izu-Bonin and Marianas.
ZONE_WHOLE_MARGIN = {
    "kermadectonga2": "Hikurangi-Tonga-Kermadec Whole margin",
    "kermadectonga": "Hikurangi-Tonga-Kermadec Whole margin",
    "kurilsjapan": "Japan/Kurile Whole Margin",
    "ryuku": "Nankai/Ryukyu Whole Margin",
    "sunda": "Andaman-Sunda Trench Whole Margin",
    "sunda2": "Andaman-Sunda Trench Whole Margin",
    "newhebrides": "New Hebrides Whole Margin",
    "newhebrides2": "New Hebrides Whole Margin",
    "solomon": "Solomons Whole margin",
    "solomon2": "Solomons Whole margin",
    "southamerica": "S. America Whole Margin",
    "hellenic2": "Hellenic Tr. Whole Margin",
}


def berryman_coupling(zone):
    """[cmin, cpref, cmax] for the zone: Berryman's Whole Margin row when
    Table 3.1 has one for it (ZONE_WHOLE_MARGIN, v9), otherwise the mean
    across this zone's merged segments, or GENERIC_COUPLING_RANGE (Berryman's
    own default for zones it does not treat, as PTHA18 applies it) if `zone`
    has no Berryman mapping."""
    whole = ZONE_WHOLE_MARGIN.get(zone)
    if whole is not None:
        r = BERRYMAN_ROWS[whole]
        return [r["coupling_min"], r["coupling_pref"], r["coupling_max"]]
    rows = _segments(zone)
    if not rows:
        return list(GENERIC_COUPLING_RANGE)
    cmin = sum(r["coupling_min"] for r in rows) / len(rows)
    cpref = sum(r["coupling_pref"] for r in rows) / len(rows)
    cmax = sum(r["coupling_max"] for r in rows) / len(rows)
    return [round(cmin, 4), round(cpref, 4), round(cmax, 4)]


def berryman_b_anchor(zone):
    """[bmin, bpref, bmax] averaged across this zone's merged segments, or
    GENERIC_B_ANCHOR (the same range most "no differentiation" Berryman
    rows already carry, see above) if `zone` has no Berryman mapping."""
    rows = _segments(zone)
    if not rows:
        return list(GENERIC_B_ANCHOR)
    bmin = sum(r["b_min"] for r in rows) / len(rows)
    bpref = sum(r["b_pref"] for r in rows) / len(rows)
    bmax = sum(r["b_max"] for r in rows) / len(rows)
    return [round(bmin, 4), round(bpref, 4), round(bmax, 4)]


def berryman_mw_max_observed(zone):
    """The largest Mmax-min of the zone's Berryman rows (its segment rows and,
    if it has one, its Whole Margin row): the largest earthquake thought to
    have occurred anywhere on the zone (ReportPTHA Section 3.7.2.1, "the
    largest earthquake thought to have occurred on the source-zone ... derived
    from Berryman et al. (2015) and the Global CMT and ISC-GEM catalogues").
    Step 6 then takes the larger of this and the zone's own GCMT maximum.

    v9 (2026-09-30). v8 returned the DOMINANT segment's Mmax-min, which is not
    the zone's largest event: sunda2 got 7.8 (Java) although its own GCMT
    catalogue holds the 2004 Sumatra-Andaman earthquake (Mw 9.03).

    If `zone` has no Berryman mapping there is no generic default (this is
    genuinely zone-specific historical data): returns None; the caller must
    supply mw_max_observed itself.
    """
    names = list(ZONE_TO_BERRYMAN_SEGMENTS.get(zone, []))
    if zone in ZONE_WHOLE_MARGIN:
        names.append(ZONE_WHOLE_MARGIN[zone])
    if not names:
        return None
    return max(BERRYMAN_ROWS[n]["mw_max_min"] for n in names)


# v9 (LEVEL 0): which Berryman et al. (2015) Table 3.1 row(s) each Bird
# (2003) plate pair of a zone covers. PTHA18 gives every segment its own
# Berryman row (sourcezone_parameters.csv: kermadectonga2_tonga 0.1/0.2/0.7
# and Mmax-min 8.0 is Berryman's "Tonga", _kermadec is "Kermadec",
# _hikurangi is "H-K-T Hikurangi", kurilsjapan_kurils is "Kurile-Kamchatka",
# _japan "Japan", ryuku_nankai "Nankai/Ryukyu Nankai", ...). A Bird segment
# can take the same row wherever its plates name the same stretch of trench
# (TO = Tonga plate; KE = Kermadec plate, which runs on under Hikurangi; OK/PA
# Kuril-Kamchatka and PA\OK the Japan Trench; PS/PA Izu-Bonin and MA/PA the
# Mariana plate; AM/PS Nankai and ON/PS the Okinawa (Ryukyu) plate). All of
# this is public: Bird's plate codes and Berryman's segment names. A Bird
# segment spanning two Berryman segments (kermadectonga2's KE/PA covers
# Kermadec and Hikurangi) takes both rows. Zones whose plate pairs do not
# name their Berryman segments (sunda2, southamerica, alaskaaleutians) are
# left out rather than guessed: their segments keep the zone's values.
BIRD_SEGMENT_TO_BERRYMAN = {
    "kermadectonga2": {"TO/PA": ["Tonga"], "KE/PA": ["Kermadec", "H-K-T Hikurangi"]},
    "kurilsjapan": {"OK/PA": ["Kurile-Kamchatka"], "PA\\OK": ["Japan"]},
    "izumariana": {"PS/PA": ["Izu-Bonin"], "MA/PA": ["Marianas"]},
    "ryuku": {"AM/PS": ["Nankai/Ryukyu Nankai"], "ON/PS": ["Nankai/Ryukyu Ryukyu"]},
}


# v9 (LEVEL 0): where each Berryman et al. (2015) segment lies on its trench.
# Table 3.1 gives, for every row, the trench end points of the segment
# (Left_E_LONG, Left_N_LAT, Right_E_LONG, Right_N_LAT; Appendix A: "the
# longitude/latitude of the left- and right-hand sides of the trench for the
# segment in this row"). Transcribed from the rendered pages 10-13 of the
# report (the text layer scrambles the Left_E_LONG column), and checked two
# ways: every interior end point equals its neighbour's (Tonga starts where
# Kermadec ends, -174.985 -23.750), and the segments' end points equal their
# Whole Margin row's (Hikurangi-Tonga-Kermadec 175.503 -42.059 to -173.407
# -14.584). Berryman places segment ends "largely where a change in plate
# motion rate and azimuth" occurs, "due to a change in the plate pairs"
# (Appendix A), mostly Bird (2003)'s, which is why Bird-derived boundaries
# land close to these. PTHA18's own segment boundaries are these ones.
BERRYMAN_SEGMENT_TRENCH = {
    # name: ((left lon, left lat), (right lon, right lat))
    "Alaska/Aleutians Komandorski": ((164.066, 55.209), (170.700, 52.498)),
    "Alaska/Aleutians Western Aleutians": ((170.700, 52.498), (-162.413, 53.367)),
    "Alaska/Aleutians Shumagin": ((-162.413, 53.367), (-157.986, 54.101)),
    "Alaska/Aleutians Semidi": ((-157.986, 54.101), (-154.160, 55.239)),
    "Alaska/Aleutians Kodiak": ((-154.160, 55.239), (-149.220, 56.925)),
    "Alaska/Aleutians Prince William Sound": ((-149.220, 56.925), (-144.316, 59.918)),
    "Alaska/Aleutians Yakataga": ((-144.316, 59.918), (-140.128, 60.381)),
    "Hellenic western segment": ((19.912, 37.731), (25.288, 34.202)),
    "Hellenic eastern segment": ((25.228, 34.202), (28.726, 36.579)),
    "Japan": ((141.992, 34.666), (144.454, 40.847)),
    # (Hellenic segments above: used by hellenic2 --segmented true, and by
    # ZONE_DEFAULT_TRENCH_ENDS below for hellenic_west2 / hellenic_east2 /
    # hellenic2. Checked against the rendered Table 3.1 page, rows 47-49.)
    "Kurile-Kamchatka": ((144.454, 40.847), (164.066, 55.209)),
    "Nankai/Ryukyu Ryukyu": ((122.501, 23.643), (132.824, 30.754)),
    "Nankai/Ryukyu Nankai": ((132.824, 30.754), (138.674, 35.034)),
    "Izu-Bonin": ((143.522, 24.391), (141.883, 34.213)),
    "Marianas": ((143.503, 11.494), (143.522, 24.391)),
    "H-K-T Hikurangi": ((175.503, -42.059), (179.838, -37.476)),
    "Kermadec": ((179.838, -37.476), (-174.985, -23.750)),
    "Tonga": ((-174.985, -23.750), (-173.407, -14.584)),
    "New Britain": ((147.283, -7.000), (153.083, -5.750)),
    "Solomon Northwest": ((153.083, -5.750), (156.296, -8.174)),
    "Solomon Southeast": ((156.296, -8.174), (164.612, -10.892)),
    "New Hebrides North": ((164.612, -10.892), (166.106, -13.634)),
    "New Hebrides Central": ((166.106, -13.634), (167.350, -18.022)),
    "New Hebrides South": ((167.350, -18.022), (169.954, -22.325)),
    "New Hebrides Matthew-Hunter": ((169.954, -22.325), (174.277, -22.667)),
    "New Guinea Trench East": ((143.743, -3.200), (138.793, -1.159)),
    "New Guinea Trench West": ((138.793, -1.159), (132.515, 0.017)),
    "Manus East": ((154.955, -4.550), (149.270, -0.650)),
    "Manus West": ((149.270, -0.650), (142.246, -2.693)),
    "Andaman-Sunda Andaman": ((92.068, 13.715), (96.202, 1.345)),
    "Andaman-Sunda Sumatra": ((96.202, 1.345), (104.576, -8.167)),
    "Andaman-Sunda Java": ((104.576, -8.167), (120.886, -11.493)),
    "S. America Ecuador-Colombia": ((-78.646, 7.337), (-81.599, -3.245)),
    "S. America Peru": ((-81.599, -3.245), (-71.307, -21.965)),
    "S. America N. Chile": ((-71.307, -21.965), (-73.246, -34.290)),
    "S. America Central Chile": ((-73.246, -34.290), (-76.006, -45.659)),
    "Patagonia North": ((-76.006, -45.659), (-76.483, -52.068)),
    "Patagonia South": ((-76.483, -52.068), (-56.925, -60.565)),
    "Mexico/CA Jalisco": ((-106.890, 21.799), (-105.247, 18.762)),
    "Mexico/CA Michoacan-Guatemala": ((-105.247, 18.762), (-90.898, 12.584)),
    "Mexico/CA El Salvador-Nicaragua": ((-90.898, 12.584), (-86.648, 10.235)),
    "Mexico/CA Costa Rica-west Panama": ((-86.648, 10.235), (-82.875, 7.366)),
}

# Default --clip (LON_MIN, LON_MAX, LAT_MIN, LAT_MAX) per zone, used by
# generate.py only when --clip is not given. Empty since 2026-10-02: the
# Hellenic zones used vertical lon lines here (19.512 / 25.258 / 29.126E, the
# segment end longitudes +0.4 deg), which cut THROUGH the slab: the cut was
# taken as trench (hellenic_east2) or left the real trench outside the window
# (hellenic_west2, whose trench runs north-south there). They are now cut at
# Berryman's own end points along down-dip lines: ZONE_DEFAULT_TRENCH_ENDS.
ZONE_DEFAULT_CLIP = {}


def _hellenic_trench_ends():
    """Ends of the Hellenic zones from Table 3.1 (rows 48, 49). The western
    segment ends at (25.288E, 34.202N) and the eastern one starts at
    (25.228E, 34.202N), 6 km apart; the two zones share the midpoint, so they
    tile with no gap and no overlap. hellenic2 (the whole arc) runs from the
    western segment's start to the eastern segment's end, where Cyprus
    begins (row 50 starts at the same 28.726E, 36.579N)."""
    w0, w1 = BERRYMAN_SEGMENT_TRENCH["Hellenic western segment"]
    e0, e1 = BERRYMAN_SEGMENT_TRENCH["Hellenic eastern segment"]
    mid = (round((w1[0] + e0[0]) / 2, 3), round((w1[1] + e0[1]) / 2, 3))
    return {"hellenic_west2": (w0, mid), "hellenic_east2": (mid, e1), "hellenic2": (w0, e1)}


# Default --trench-ends ((LON, LAT), (LON, LAT)) per zone, used by generate.py
# only when --trench-ends is not given: for a zone that is one part of a SLAB
# region, step 1 builds the trench, datum and contours on the whole raster and
# cuts every contour along the down-dip line through each end point
# (slab_contours.below_trench_contours(trench_ends=)).
ZONE_DEFAULT_TRENCH_ENDS = _hellenic_trench_ends()


def zone_plate_pairs(zone):
    """Berryman's plate pairs of a zone defined by Berryman's segments
    (the ZONE_DEFAULT_TRENCH_ENDS zones), for step 3 to match only Bird
    steps between those plates (bird_convergence.column_convergence's
    plate_pairs). None for every other zone: they keep the plain nearest
    Bird step, as PTHA18 did."""
    if zone not in ZONE_DEFAULT_TRENCH_ENDS:
        return None
    rows = [r for _, r in ZONE_BERRYMAN_SEGMENTS.get(zone, [])] or ZONE_TO_BERRYMAN_SEGMENTS.get(zone, [])
    return sorted({BERRYMAN_SEGMENT_PLATES[r] for r in rows}) or None

# Table 3.1's "Plate pairs" of the same rows (overriding\subducting, Bird's
# plate codes except HF, the Hikurangi forearc of Wallace et al. 2004).
BERRYMAN_SEGMENT_PLATES = {
    **{f"Alaska/Aleutians {s}": "PA\\NA" for s in (
        "Komandorski", "Western Aleutians", "Shumagin", "Semidi", "Kodiak",
        "Prince William Sound", "Yakataga")},
    "Japan": "PA\\OK", "Kurile-Kamchatka": "PA\\OK",
    "Nankai/Ryukyu Ryukyu": "PS\\ON", "Nankai/Ryukyu Nankai": "PS\\AM",
    "Izu-Bonin": "PA\\PS", "Marianas": "PA\\MA",
    "H-K-T Hikurangi": "PA\\HF", "Kermadec": "PA\\KE", "Tonga": "PA\\TO",
    "New Britain": "WL\\SB", "Solomon Northwest": "WL\\PA", "Solomon Southeast": "AU\\PA",
    "New Hebrides North": "AU\\PA", "New Hebrides Central": "AU\\NH",
    "New Hebrides South": "AU\\NH", "New Hebrides Matthew-Hunter": "AU\\MH",
    "New Guinea Trench East": "PA\\NGH", "New Guinea Trench West": "CL\\BH",
    "Manus East": "PA\\NB", "Manus West": "CL\\NB",
    "Andaman-Sunda Andaman": "IN or AU\\BU", "Andaman-Sunda Sumatra": "AU\\SU",
    "Andaman-Sunda Java": "AU\\SU",
    "S. America Ecuador-Colombia": "NZ\\ND", "S. America Peru": "NZ\\SA or AP",
    "S. America N. Chile": "NZ\\SA", "S. America Central Chile": "NZ\\SA",
    "Patagonia North": "AN\\SA", "Patagonia South": "AN\\SC",
    "Hellenic western segment": "AF\\AS", "Hellenic eastern segment": "AF\\AS",
    "Mexico/CA Jalisco": "RI\\NA", "Mexico/CA Michoacan-Guatemala": "CO\\NA",
    "Mexico/CA El Salvador-Nicaragua": "CO\\CA", "Mexico/CA Costa Rica-west Panama": "CO\\PM",
}

# v9 (LEVEL 0): the segments Berryman et al. (2015) divide each zone into, as
# (segment key, Table 3.1 row). The keys follow PTHA18's segment names where
# PTHA18 has the same segment, so the two runs' tables line up by name.
# Rows are the zone's stretch of Table 3.1; a mesh that does not reach a
# segment simply gives it no columns (segmentation.berryman_segments drops
# it). southamerica includes Berryman's Patagonia rows, as PTHA18's zone does
# (_patagonia); solomon includes New Britain, as PTHA18's does (_newbritain).
ZONE_BERRYMAN_SEGMENTS = {
    "kermadectonga2": [("hikurangi", "H-K-T Hikurangi"), ("kermadec", "Kermadec"),
                       ("tonga", "Tonga")],
    "kurilsjapan": [("japan", "Japan"), ("kurils", "Kurile-Kamchatka")],
    "izumariana": [("marianas", "Marianas"), ("izubonin", "Izu-Bonin")],
    "ryuku": [("ryuku", "Nankai/Ryukyu Ryukyu"), ("nankai", "Nankai/Ryukyu Nankai")],
    "sunda2": [("andaman", "Andaman-Sunda Andaman"), ("sumatra", "Andaman-Sunda Sumatra"),
               ("java", "Andaman-Sunda Java")],
    "southamerica": [("ecuador", "S. America Ecuador-Colombia"), ("peru", "S. America Peru"),
                     ("northern_chile", "S. America N. Chile"),
                     ("central_chile", "S. America Central Chile"),
                     ("patagonia_north", "Patagonia North"),
                     ("patagonia_south", "Patagonia South")],
    "alaskaaleutians": [("komandorski", "Alaska/Aleutians Komandorski"),
                        ("western", "Alaska/Aleutians Western Aleutians"),
                        ("shumagin", "Alaska/Aleutians Shumagin"),
                        ("semidi", "Alaska/Aleutians Semidi"),
                        ("kodiak", "Alaska/Aleutians Kodiak"),
                        ("prince_william_sound", "Alaska/Aleutians Prince William Sound"),
                        ("yakataga", "Alaska/Aleutians Yakataga")],
    "newhebrides2": [("north", "New Hebrides North"), ("central", "New Hebrides Central"),
                     ("south", "New Hebrides South"),
                     ("matthewhunter", "New Hebrides Matthew-Hunter")],
    "solomon2": [("newbritain", "New Britain"), ("northwest", "Solomon Northwest"),
                 ("southeast", "Solomon Southeast")],
    "mexico": [("jalisco", "Mexico/CA Jalisco"),
               ("michoacan_guatemala", "Mexico/CA Michoacan-Guatemala"),
               ("elsalvador_nicaragua", "Mexico/CA El Salvador-Nicaragua"),
               ("costarica_panama", "Mexico/CA Costa Rica-west Panama")],
    "newguinea2": [("east", "New Guinea Trench East"), ("west", "New Guinea Trench West")],
    "manus": [("east", "Manus East"), ("west", "Manus West")],
    "hellenic2": [("west", "Hellenic western segment"), ("east", "Hellenic eastern segment")],
}
for _z in ("kermadectonga", "sunda", "newhebrides", "solomon", "newguinea"):
    ZONE_BERRYMAN_SEGMENTS[_z] = ZONE_BERRYMAN_SEGMENTS[_z + "2"]


def berryman_rows_for_bird_segment(zone, plate_pair):
    """v9: the Berryman row names a Bird segment of `zone` takes (see
    BIRD_SEGMENT_TO_BERRYMAN), or None when the table has no entry."""
    return BIRD_SEGMENT_TO_BERRYMAN.get(zone, {}).get(plate_pair)


def berryman_segment_coupling(rows):
    """[cmin, cpref, cmax] of the given Berryman rows (their mean when a Bird
    segment spans more than one, the rule berryman_coupling uses for a zone)."""
    rr = [BERRYMAN_ROWS[r] for r in rows]
    return [round(sum(r[k] for r in rr) / len(rr), 4)
            for k in ("coupling_min", "coupling_pref", "coupling_max")]


def berryman_segment_mw_max_min(rows):
    """The largest Mmax-min of the given rows: the largest earthquake thought
    to have occurred on that stretch (ReportPTHA Section 3.7.2.1)."""
    return max(BERRYMAN_ROWS[r]["mw_max_min"] for r in rows)


def berryman_segment_mw_max_floor(zone):
    """v9 (LEVEL 0): the smallest Mmax-min Berryman et al. (2015) give ANY of
    the zone's segments, used as a floor for every Bird segment's
    mw_max_observed.

    Bird's segments and Berryman's are not the same pieces, and Berryman's
    table has no geography to match them by, so no single Berryman row can be
    assigned to a Bird segment. The smallest value is the one every stretch of
    the zone is known to reach (kurilsjapan: Japan 9.0 and Kurile-Kamchatka
    9.0 -> 9.0; kermadectonga2: Kermadec 8.1 and Tonga 8.0 -> 8.0). It keeps
    a segment whose largest event predates GCMT (Kamchatka 1952, Mw 9.0) from
    being modelled as if nothing that large had happened there.

    Returns None when the zone has no Berryman mapping.
    """
    rows = ZONE_TO_BERRYMAN_SEGMENTS.get(zone)
    if not rows:
        return None
    return min(BERRYMAN_ROWS[r]["mw_max_min"] for r in rows)
