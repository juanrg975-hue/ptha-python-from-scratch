"""Replication of rptha's OFFICIAL testthat suite for the ported functions.

Every test here is transcribed from rptha/R/rptha/tests/testthat/*.R (the package's
own test files, NOT invented cases). Where the original test relies on
functions pyptha does not port (dGR/rGR sampling, kajiura, interior points,
ncdf4 I/O) it is not replicated; see the module-level notes at the bottom.

Sources:
    test_mean_angle.R
    test_distance_down_depth.R
    test_interpolate_gc_path.R
    test_spherical_to_cartesian2d_and_inverse.R
    test_rupture_creation_and_probabilities.R  (Alaska shapefile pipeline)

Run:  python -m pytest pyptha/tests/test_official_rptha_suite.py -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pyptha_v12 import contour_discretisation as cd  # noqa: E402
from pyptha_v12 import events, rates, unit_sources as us  # noqa: E402
from pyptha_v12.unit_sources import _mean_angle, distance_down_depth  # noqa: E402

_ALASKA_SHP = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "rptha", "R", "rptha", "tests", "testthat", "testshp", "alaska.shp"))


def _shp_available():
    try:
        import geopandas  # noqa: F401
    except Exception:
        return False
    return os.path.exists(_ALASKA_SHP)


# ---------------------------------------------------------------------------
# test_mean_angle.R
# ---------------------------------------------------------------------------

def test_mean_angle_official():
    # By this method the mean of 0,0,90 is < 30 (golden: 26.5650511771)
    m1 = np.degrees(_mean_angle(np.radians([0.0, 0.0, 90.0])))
    assert abs(m1 - 26.5650511771) < 1.0e-08

    # A more typical case
    m2 = np.degrees(_mean_angle(np.radians([120.0, 60.0])))
    assert abs(m2 - 90.0) < np.sqrt(np.finfo(float).eps)

    # Negative angles represented as positive give the same mean
    rng = np.random.default_rng(42)
    a3 = rng.uniform(-2, 2, 100) * np.pi
    a5 = a3 + (a3 < 0) * 2 * np.pi
    m3 = _mean_angle(a3)
    m5 = _mean_angle(a5)
    # compare directions on the circle (means can differ by 2*pi)
    assert abs(np.exp(1j * m3) - np.exp(1j * m5)) < 1.0e-06


# ---------------------------------------------------------------------------
# test_distance_down_depth.R
# ---------------------------------------------------------------------------

def test_distance_down_depth_official():
    # TEST 1: surface distance equals great-circle distance
    # (R checks against geosphere::distCosine on a sphere R=6378137; our port
    # uses the same spherical model.)
    p1 = [200.0, 50.0, 0.0]
    p2 = [170.0, 10.0, 0.0]
    dist1 = distance_down_depth(p2, p1, n=int(1e4))
    # Great-circle on the sphere used by the port
    R = 6378137.0
    lon1, lat1 = np.radians([p2[0], p2[1]])
    lon2, lat2 = np.radians([p1[0], p1[1]])
    dist_cos = R * np.arccos(np.sin(lat1) * np.sin(lat2)
                             + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))
    assert abs(dist1 - dist_cos) / dist_cos < 1e-6

    # TEST 2: unaffected by -360 shift in longitude
    d2 = distance_down_depth([170.0, 10.0, 50.0], [200.0, 50.0, 0.0])
    d3 = distance_down_depth([170.0, 10.0, 50.0], [200.0 - 360.0, 50.0, 0.0])
    assert np.isclose(d2, d3)

    # TEST 3: symmetric in its arguments
    d4 = distance_down_depth([200.0 - 360.0, 50.0, 0.0], [170.0, 10.0, 50.0])
    assert np.isclose(d3, d4)

    # TEST 4: unaffected by reflecting latitude
    d5 = distance_down_depth([170.0, -10.0, 50.0], [200.0, -50.0, 0.0])
    assert np.isclose(d5, d3)

    # TEST 5: analytical solution for a 1D dipping fault (drop 50 km in 1 deg)
    from scipy.integrate import quad
    Rkm = 6378.137
    ddepth_dtheta = 50.0 / (2 * np.pi / 360.0)

    def f(theta):
        return np.sqrt(ddepth_dtheta ** 2 + (Rkm - theta * ddepth_dtheta) ** 2)

    l_analytical, _ = quad(f, 0.0, 2 * np.pi / 360.0)
    l_numerical = distance_down_depth([0.0, 0.0, 0.0], [1.0, 0.0, 50.0]) / 1000.0
    assert abs(l_analytical - l_numerical) / l_analytical < 1e-6


# ---------------------------------------------------------------------------
# test_interpolate_gc_path.R
# ---------------------------------------------------------------------------

def test_interpolate_gc_path_official():
    # Test 1
    surface_path = np.array([[200.0, 5.0], [170.0, -5.0]])
    ip1 = cd.interpolate_gc_path(surface_path)
    assert np.all(np.isfinite(ip1))
    assert np.max(np.abs(np.diff(ip1[:, 0]))) < 20

    # Test 2: reversed path retraces the same points
    ip2 = cd.interpolate_gc_path(surface_path[::-1])
    assert np.max(np.abs(np.diff(ip2[:, 0]))) < 2
    assert not np.any((np.abs(ip2[::-1, 0] - ip1[:, 0]) > 1.0e-03)
                      | (np.abs(ip2[::-1, 1] - ip1[:, 1]) > 1.0e-03))

    # Test 3: crossing the dateline stays continuous
    surface_path = np.array([[-160.0, 5.0], [170.0, -5.0]])
    ip1 = cd.interpolate_gc_path(surface_path)
    assert np.all(np.isfinite(ip1))
    assert np.max(np.abs(np.diff(ip1[:, 0]))) < 20

    # Test 4: reversed dateline path same modulo 360
    ip2 = cd.interpolate_gc_path(surface_path[::-1])
    assert np.max(np.abs(np.diff(ip2[:, 0]))) < 2
    assert not np.any((np.abs(ip2[::-1, 0] - ip1[:, 0]) % 360 > 1.0e-03)
                      | (np.abs(ip2[::-1, 1] - ip1[:, 1]) > 1.0e-03))


# ---------------------------------------------------------------------------
# test_spherical_to_cartesian2d_and_inverse.R (forward + distance checks;
# pyptha has no inverse function so Test 1's roundtrip uses a local inverse)
# ---------------------------------------------------------------------------

def test_spherical_to_cartesian2d_official():
    rng = np.random.default_rng(1234)
    lonlat = np.column_stack([20 + 0.1 * rng.uniform(size=10),
                              -19 + 0.1 * rng.uniform(size=10)])
    origin = np.array([20.0, -19.0])

    new_coords = cd._spherical_to_cartesian2d(lonlat, origin)
    # invert the (deg -> local metres) projection analytically
    R = 6378137.0
    lon_back = new_coords[:, 0] / (R * np.cos(np.radians(origin[1]))) / np.pi * 180 + origin[0]
    lat_back = new_coords[:, 1] / R / np.pi * 180 + origin[1]
    assert np.all(np.abs(lonlat[:, 0] - lon_back) < 1.0e-06)
    assert np.all(np.abs(lonlat[:, 1] - lat_back) < 1.0e-06)

    # Test 3 (distance check vs spherical haversine, tolerance as in R)
    for origin, tol in (([144.96, -37.81], 0.02), ([0.0, 62.0], 0.04)):
        origin = np.array(origin)
        pts = np.column_stack([origin[0] + rng.uniform(size=40),
                               origin[1] + rng.uniform(size=40)])
        xy = cd._spherical_to_cartesian2d(pts, origin)
        d_local = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        lam = np.radians(pts[:, 0])
        phi = np.radians(pts[:, 1])
        dphi = phi[:, None] - phi[None, :]
        dlam = lam[:, None] - lam[None, :]
        a = (np.sin(dphi / 2) ** 2
             + np.cos(phi)[:, None] * np.cos(phi)[None, :] * np.sin(dlam / 2) ** 2)
        d_sph = 2 * R * np.arcsin(np.sqrt(a))
        mask = d_local > 0
        assert np.max(np.abs(d_sph[mask] - d_local[mask]) / d_local[mask]) < tol


# ---------------------------------------------------------------------------
# test_rupture_creation_and_probabilities.R (Alaska pipeline)
# ---------------------------------------------------------------------------

def _load_alaska():
    import geopandas as gpd
    gdf = gpd.read_file(_ALASKA_SHP)
    gdf["level"] = gdf["level"].astype(float)
    out = []
    for _, row in gdf.iterrows():
        g = row.geometry
        coords = (np.array(g.coords) if g.geom_type == "LineString"
                  else np.array(sorted(g.geoms, key=lambda x: len(x.coords))[-1].coords))
        out.append((row["level"], coords[:, :2]))
    return out


def _alaska_setup():
    """Discretise Alaska (100x50 km cells, as the official test) and build the
    event table + conditional probabilities exactly as the R test does."""
    contours = _load_alaska()
    grid = cd.discretized_source_from_contours_orthogonal(
        contours, desired_unit_source_length=100.0,
        desired_unit_source_width=50.0, seed=1234)
    stats = us.discretized_source_approximate_summary_statistics(grid)
    dMw = 0.1
    eq_table = events.get_all_earthquake_events(
        stats, Mmin=7.5, Mmax=9.4, dMw=dMw)
    et = {"Mw": eq_table["Mw"], "slip": eq_table["slip"], "area": eq_table["area"]}
    ecp = rates.get_event_probabilities_conditional_on_Mw(et, "inverse_slip")
    area = float(np.sum(stats["length"] * stats["width"]))
    return stats, eq_table, et, ecp, area, dMw


def test_alaska_event_index_string_roundtrip():
    """R lines 30-40: event_index_string decodes back to the event's indices."""
    if not _shp_available():
        print("SKIP (geopandas or alaska.shp unavailable)")
        return
    _, eq_table, _, _, _, _ = _alaska_setup()
    n = len(eq_table["event_index_string"])
    ei = int(np.ceil(n / 2)) - 1
    idx = events.get_unit_source_indices_in_event(eq_table["event_index_string"][ei])
    back = "".join(f"{i + 1}-" for i in idx)
    assert back == eq_table["event_index_string"][ei]


def test_alaska_rate_gt9_in_600_700():
    """R lines 48-96: with the official recurrence parameters, the return
    period of Mw>9 lies in (600, 700) years, and accounting for moment below
    Mw_min lowers the rate."""
    if not _shp_available():
        print("SKIP (geopandas or alaska.shp unavailable)")
        return
    _, _, et, ecp, area, dMw = _alaska_setup()

    kwargs = dict(
        slip_rate=np.array([44.00, 49.50, 55.00]) / 1000,
        slip_rate_prob=[0.6, 0.2, 0.2],
        b=[0.95, 0.7, 1.2], b_prob=[0.6, 0.2, 0.2],
        Mw_min=[7.5 - dMw / 2], Mw_min_prob=[1.0],
        Mw_max=[9.40, 9.40, 9.00], Mw_max_prob=[0.6, 0.3, 0.1],
        sourcezone_total_area=area,
        event_table=et, event_conditional_probabilities=ecp)

    rate_fn = rates.rate_of_earthquakes_greater_than_Mw_function(**kwargs)
    freq_gt9 = 1.0 / float(rate_fn(9.0))
    assert 600 < freq_gt9 < 700, f"RP(Mw>9) = {freq_gt9:.1f} not in (600, 700)"

    rate_fnB = rates.rate_of_earthquakes_greater_than_Mw_function(
        account_for_moment_below_mwmin=True, **kwargs)
    freq_gt9B = 1.0 / float(rate_fnB(9.0))
    assert freq_gt9 < freq_gt9B

    # R lines 144-156: mean of the 0.1..0.9 quantiles ~ the weighted-mean rate.
    # pyptha's quantile output is (len(quantiles), len(Mw)); for scalar Mw the
    # Mw axis has length 1 (squeezable), vs R's (len(Mw), len(quantiles)).
    q = np.ravel(rate_fn(9.0, quantiles=np.arange(0.1, 0.91, 0.1)))
    assert abs(np.mean(q) - float(rate_fn(9.0))) < 1e-4
    # multiple Mw and quantiles at once agree with one-at-a-time
    q_multi = rate_fn(np.array([9.0, 9.0, 9.1]), quantiles=np.arange(0.1, 0.91, 0.1))
    q91 = np.ravel(rate_fn(9.1, quantiles=np.arange(0.1, 0.91, 0.1)))
    assert np.all(q == q_multi[:, 0])
    assert np.all(q == q_multi[:, 1])
    assert np.all(q91 == q_multi[:, 2])


def test_alaska_longterm_slip_balance():
    """R lines 165-211: back-calculated long-term slip matches the input
    slip rate (unweighted within 1%, area-weighted within 1e-8)."""
    if not _shp_available():
        print("SKIP (geopandas or alaska.shp unavailable)")
        return
    stats, eq_table, et, ecp, area, dMw = _alaska_setup()

    slip_rate = np.array([44.00, 49.50, 55.00]) / 1000
    slip_rate_prob = np.array([0.6, 0.2, 0.2])
    rate_fn = rates.rate_of_earthquakes_greater_than_Mw_function(
        slip_rate=slip_rate, slip_rate_prob=slip_rate_prob,
        b=[0.95, 0.7, 1.2], b_prob=[0.6, 0.2, 0.2],
        Mw_min=[7.5 - dMw / 2], Mw_min_prob=[1.0],
        Mw_max=[9.40, 9.40, 9.00], Mw_max_prob=[0.6, 0.3, 0.1],
        sourcezone_total_area=area,
        event_table=et, event_conditional_probabilities=ecp)

    mw = eq_table["Mw"]
    event_rate = ecp * (rate_fn(mw - dMw / 2) - rate_fn(mw + dMw / 2))
    ev_slip = eq_table["slip"] * event_rate
    ev_slip_area = ev_slip * eq_table["area"]

    n_us = stats["subfault_number"].size
    us_slip = np.zeros(n_us)
    us_slip_area = np.zeros(n_us)
    for ee in range(mw.size):
        idx = events.get_unit_source_indices_in_event(
            eq_table["event_index_string"][ee])
        us_slip[idx] += ev_slip[ee]
        us_slip_area[idx] += ev_slip_area[ee]

    unit_area = stats["length"] * stats["width"]
    theoretical = float(np.sum(slip_rate * slip_rate_prob))
    assert abs(us_slip.mean() - theoretical) / theoretical < 0.01
    weighted = float(np.average(us_slip, weights=unit_area))
    assert abs(weighted - theoretical) / theoretical < 1.0e-08


def test_alaska_logic_tree_weighting():
    """R lines 219-276: the combined-b curve equals the probability-weighted
    sum of the single-b curves, exactly."""
    if not _shp_available():
        print("SKIP (geopandas or alaska.shp unavailable)")
        return
    _, _, et, ecp, area, _ = _alaska_setup()

    common = dict(slip_rate=[55.00 / 1000], slip_rate_prob=[1.0],
                  Mw_min=[7.5], Mw_min_prob=[1.0],
                  Mw_max=[9.40], Mw_max_prob=[1.0],
                  sourcezone_total_area=area,
                  event_table=et, event_conditional_probabilities=ecp)
    b, b_prob = [0.7, 1.2], [0.3, 0.7]

    fn_mix = rates.rate_of_earthquakes_greater_than_Mw_function(
        b=b, b_prob=b_prob, **common)
    fn_07 = rates.rate_of_earthquakes_greater_than_Mw_function(
        b=[b[0]], b_prob=[1.0], **common)
    fn_12 = rates.rate_of_earthquakes_greater_than_Mw_function(
        b=[b[1]], b_prob=[1.0], **common)

    mws = np.round(np.arange(7.5, 9.51, 0.1), 6)
    err = fn_mix(mws) - (b_prob[0] * fn_07(mws) + b_prob[1] * fn_12(mws))
    np.testing.assert_allclose(err, 0.0, atol=1e-15)


def test_alaska_bayesian_data_selects_branch():
    """R lines 288-349: with data strongly favouring 100 mm/yr over 0.1 mm/yr,
    the updated mixture equals the 100 mm/yr branch to 1e-5."""
    if not _shp_available():
        print("SKIP (geopandas or alaska.shp unavailable)")
        return
    _, _, et, ecp, area, _ = _alaska_setup()

    common = dict(b=[1.0], b_prob=[1.0], Mw_min=[7.5], Mw_min_prob=[1.0],
                  Mw_max=[9.40], Mw_max_prob=[1.0],
                  sourcezone_total_area=area,
                  event_table=et, event_conditional_probabilities=ecp)

    fn_fast = rates.rate_of_earthquakes_greater_than_Mw_function(
        slip_rate=[100.0 / 1000], slip_rate_prob=[1.0], **common)
    fn_mix = rates.rate_of_earthquakes_greater_than_Mw_function(
        slip_rate=np.array([0.1, 100.0]) / 1000, slip_rate_prob=[0.5, 0.5],
        update_logic_tree_weights_with_data=True,
        Mw_count_duration=(7.6, 3, 50), **common)

    mws = np.linspace(7.5, 9.399, 100)
    np.testing.assert_allclose(fn_mix(mws), fn_fast(mws), rtol=1.0e-05)


# ---------------------------------------------------------------------------
# NOT replicated (functions not ported to pyptha):
#   test_GR.R                      dGR/pGR/qGR/rGR density family
#   test_kajiura_filter.R          Kajiura filter
#   test_unit_source_interior_points_cartesian.R   sub-unit-source grids
#   test_unit_source_cartesian_to_okada_tsunami_source.R  (same)
#   test_write_table_to_ncdf4.R    ncdf4 I/O
#   test_interpolation.R           unstructured interpolation module
#   test_intersect_surface_path_with_depth_contours.R  (legacy discretiser)
#   test_adjust_longitude_by_360_deg.R  trivial helper, not ported
#   test_okada_tsunami.R           ALREADY replicated in test_r_parity.py
#   test_rupture_scaling.R         ALREADY replicated in test_r_parity.py
#   test_discrete_source_summary_statistics.R  partially covered by
#       test_r_parity.test_summary_stats_analytic_dip (the official test's
#       exact numbers depend on rptha's full grid discretiser)
# ---------------------------------------------------------------------------


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
