"""Zone-level Berryman et al. (2015) values (v9, 2026-09-30).

Coupling is Berryman's Whole Margin row where Table 3.1 has one for the zone
(the value PTHA18 used too), the mean of the segment rows otherwise; the zone's
mw_max_observed floor is the largest Mmax-min of its rows (step 6 then takes
the larger of that and the zone's GCMT maximum). b is unchanged: the mean of
Berryman's rows.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_PKG, "lib"))

import berryman_params as bp  # noqa: E402


def test_coupling_is_the_whole_margin_row_where_there_is_one():
    assert bp.berryman_coupling("kermadectonga2") == [0.21, 0.31, 0.72]
    assert bp.berryman_coupling("kurilsjapan") == [0.67, 0.77, 0.9]
    assert bp.berryman_coupling("ryuku") == [0.34, 0.44, 0.8]
    for zone, row in bp.ZONE_WHOLE_MARGIN.items():
        assert row in bp.BERRYMAN_ROWS, (zone, row)
    # no Whole Margin row: the mean of Izu-Bonin and Marianas, as before
    assert "izumariana" not in bp.ZONE_WHOLE_MARGIN
    assert bp.berryman_coupling("izumariana") == [0.1, 0.2, 0.7]


def test_mw_max_observed_is_the_largest_row_not_the_dominant_one():
    # sunda2's dominant segment (Java) has 7.8; the zone hosted Sumatra 2004
    assert bp.berryman_mw_max_observed("sunda2") == 9.0
    assert bp.berryman_mw_max_observed("kermadectonga2") == 8.1
    assert bp.berryman_mw_max_observed("ryuku") == 8.5
    assert bp.berryman_mw_max_observed("not-a-zone") is None
    # b is untouched (the mean of the zone's Berryman rows)
    assert bp.berryman_b_anchor("kurilsjapan") == [0.62, 0.915, 1.2]


def test_hellenic_zones_match_only_their_own_bird_plates():
    """v9: hellenic_west2 / hellenic_east2 / hellenic2 match only Bird steps
    between Berryman's plates (AF-AS = Bird AS/AF); every other zone keeps
    the plain nearest Bird step (zone_plate_pairs is None)."""
    import berryman_params as bp
    import bird_convergence as bc
    for z in ("hellenic_west2", "hellenic_east2", "hellenic2"):
        pairs = bp.zone_plate_pairs(z)
        assert pairs and {bc.plate_codes(p) for p in pairs} == {bc.plate_codes("AS/AF")}
    for z in ("kermadectonga2", "kurilsjapan", "calabria2", "antilles2", "southamerica"):
        assert bp.zone_plate_pairs(z) is None
    assert bc.plate_codes("EU-AF") != bc.plate_codes("AS/AF")
