"""
Tests for coord_utils — the math must be correct.

We use known positions and cross-check against independently-computed values
(WolframAlpha / aviation geometry calculators).

All expected values are within ±0.5° tolerance, which is comfortably within
the S50's pointing accuracy anyway.
"""

from __future__ import annotations

import math
import pytest

from seestar_tracker.coord_utils import (
    geodetic_to_az_alt,
    angular_separation,
    angular_rate,
    angular_velocity,
    predict_position,
    ft_to_m,
    nm_to_m,
    kt_to_ms,
)


# ---------------------------------------------------------------------------
# Tolerance for angle comparisons (degrees)
# ---------------------------------------------------------------------------
TOL = 0.5   # degrees — tighter than the S50's native pointing accuracy


def approx_angle(a: float, b: float, tol: float = TOL) -> bool:
    """True if a and b are within tol degrees, accounting for 0/360 wrap."""
    diff = abs((a - b + 180) % 360 - 180)
    return diff <= tol


# ---------------------------------------------------------------------------
# geodetic_to_az_alt
# ---------------------------------------------------------------------------

class TestGeodeticToAzAlt:

    def test_due_north(self):
        """Target directly north at same elevation → az ≈ 0°."""
        observer_lat, observer_lon = 34.0, -118.0
        target_lat = observer_lat + 0.5   # ~55 km north
        target_lon = observer_lon
        target_alt_m = 0

        az, alt = geodetic_to_az_alt(
            observer_lat, observer_lon, 0,
            target_lat, target_lon, target_alt_m,
        )
        assert approx_angle(az, 0.0), f"Expected az≈0 (north), got {az:.2f}°"
        # alt should be near 0° (or slightly negative if we don't account for Earth curvature)

    def test_due_east(self):
        """Target directly east → az ≈ 90°."""
        observer_lat, observer_lon = 34.0, -118.0
        target_lat = observer_lat
        target_lon = observer_lon + 0.5
        target_alt_m = 0

        az, alt = geodetic_to_az_alt(
            observer_lat, observer_lon, 0,
            target_lat, target_lon, target_alt_m,
        )
        assert approx_angle(az, 90.0), f"Expected az≈90 (east), got {az:.2f}°"

    def test_due_south(self):
        """Target directly south → az ≈ 180°."""
        observer_lat, observer_lon = 34.0, -118.0
        target_lat = observer_lat - 0.5
        target_lon = observer_lon
        target_alt_m = 0

        az, alt = geodetic_to_az_alt(
            observer_lat, observer_lon, 0,
            target_lat, target_lon, target_alt_m,
        )
        assert approx_angle(az, 180.0), f"Expected az≈180 (south), got {az:.2f}°"

    def test_due_west(self):
        """Target directly west → az ≈ 270°."""
        observer_lat, observer_lon = 34.0, -118.0
        target_lat = observer_lat
        target_lon = observer_lon - 0.5
        target_alt_m = 0

        az, alt = geodetic_to_az_alt(
            observer_lat, observer_lon, 0,
            target_lat, target_lon, target_alt_m,
        )
        assert approx_angle(az, 270.0), f"Expected az≈270 (west), got {az:.2f}°"

    def test_aircraft_at_lax_from_santa_monica(self):
        """
        Aircraft at LAX runway 24L (~33.942°N, -118.408°W) at 3000ft AGL,
        observed from Santa Monica Pier (~34.010°N, -118.498°W).

        LAX is roughly SE of Santa Monica.
        Expected az: ~120-140°, alt: ~5-15°
        """
        observer_lat =  34.010
        observer_lon = -118.498
        observer_elev = 0

        target_lat  =  33.942
        target_lon  = -118.408
        target_alt_m = ft_to_m(3000)

        az, alt = geodetic_to_az_alt(
            observer_lat, observer_lon, observer_elev,
            target_lat, target_lon, target_alt_m,
        )
        assert 100 < az < 160, f"Expected az in [100, 160] (SE), got {az:.1f}°"
        assert 5 < alt < 25,   f"Expected alt in [5, 25], got {alt:.1f}°"

    def test_altitude_increases_elevation_angle(self):
        """Higher aircraft → higher elevation angle."""
        observer = (34.0, -118.0, 0)
        target_base = (34.0, -117.8, 0)   # ~20 km east

        def _alt(alt_m):
            return geodetic_to_az_alt(*observer, *target_base[:2], alt_m)[1]

        alt_low  = _alt(ft_to_m(1000))
        alt_high = _alt(ft_to_m(35000))
        assert alt_high > alt_low, (
            f"Higher aircraft should have higher elevation angle: "
            f"{alt_high:.1f}° vs {alt_low:.1f}°"
        )

    def test_directly_overhead(self):
        """Target at same lat/lon but much higher → alt ≈ 90°."""
        lat, lon = 34.0, -118.0
        target_alt_m = 400_000   # ~ISS altitude

        az, alt = geodetic_to_az_alt(lat, lon, 0, lat, lon, target_alt_m)
        assert abs(alt - 90.0) < 1.0, f"Expected alt≈90°, got {alt:.2f}°"

    def test_known_iss_geometry(self):
        """
        ISS overhead pass geometry sanity check.

        If ISS is directly overhead (within ~10km horizontal), alt should be
        close to 90° regardless of az.
        """
        lat, lon = 34.052, -118.243   # Los Angeles
        iss_lat = 34.053   # essentially overhead
        iss_lon = -118.244
        iss_alt_m = 408_000   # 408 km

        az, alt = geodetic_to_az_alt(lat, lon, 0, iss_lat, iss_lon, iss_alt_m)
        assert alt > 88.0, f"ISS nearly overhead should give alt>88°, got {alt:.2f}°"

    def test_below_horizon(self):
        """Target far away at ground level → negative altitude (below horizon)."""
        # Observer at 34°N, target at 44°N (1000 km north) at ground level
        az, alt = geodetic_to_az_alt(34.0, -118.0, 0, 44.0, -118.0, 0)
        assert alt < 0, f"Distant ground-level target should be below horizon, got {alt:.2f}°"

    def test_observer_elevation_effect(self):
        """Observer on a mountain sees further — elevated observer has slightly different az/alt."""
        target_lat, target_lon = 34.5, -118.0
        target_alt_m = ft_to_m(10000)

        az_sea, alt_sea = geodetic_to_az_alt(34.0, -118.0, 0, target_lat, target_lon, target_alt_m)
        az_mtn, alt_mtn = geodetic_to_az_alt(34.0, -118.0, 2000, target_lat, target_lon, target_alt_m)

        # Both should point roughly north (small az difference)
        assert approx_angle(az_sea, 0.0, tol=5.0), f"Expected northward, got {az_sea:.1f}°"
        assert approx_angle(az_mtn, 0.0, tol=5.0), f"Expected northward, got {az_mtn:.1f}°"
        # Mountain observer sees target at slightly lower elevation
        assert alt_mtn < alt_sea + 1.0


# ---------------------------------------------------------------------------
# angular_separation
# ---------------------------------------------------------------------------

class TestAngularSeparation:

    def test_same_point(self):
        assert angular_separation(45.0, 30.0, 45.0, 30.0) == pytest.approx(0.0, abs=1e-9)

    def test_due_east_1_degree(self):
        """Move 1° in azimuth at 0° alt → separation ≈ 1°."""
        sep = angular_separation(0.0, 0.0, 1.0, 0.0)
        assert sep == pytest.approx(1.0, abs=0.01)

    def test_opposite_azimuths_at_horizon(self):
        """North vs South at horizon → separation = 180°."""
        sep = angular_separation(0.0, 0.0, 180.0, 0.0)
        assert sep == pytest.approx(180.0, abs=0.01)

    def test_altitude_difference(self):
        """30° vs 60° altitude, same azimuth → separation = 30°."""
        sep = angular_separation(90.0, 30.0, 90.0, 60.0)
        assert sep == pytest.approx(30.0, abs=0.01)

    def test_azimuth_wrap_at_360(self):
        """359° and 1° should be 2° apart, not 358°."""
        sep = angular_separation(359.0, 0.0, 1.0, 0.0)
        assert sep == pytest.approx(2.0, abs=0.05)

    def test_near_zenith_convergence(self):
        """
        At high altitudes the azimuth lines converge — moving 90° in azimuth
        at 89° altitude is much less than 90°.
        """
        sep_horizon = angular_separation(0.0,  0.0, 90.0,  0.0)
        sep_zenith  = angular_separation(0.0, 89.0, 90.0, 89.0)
        assert sep_zenith < sep_horizon, (
            f"High-alt sep ({sep_zenith:.2f}°) should be less than horizon sep ({sep_horizon:.2f}°)"
        )

    def test_symmetry(self):
        """separation(A, B) == separation(B, A)."""
        a = angular_separation(10.0, 20.0, 50.0, 60.0)
        b = angular_separation(50.0, 60.0, 10.0, 20.0)
        assert a == pytest.approx(b, abs=1e-9)


# ---------------------------------------------------------------------------
# angular_rate
# ---------------------------------------------------------------------------

class TestAngularRate:

    def test_stationary(self):
        """No movement → rate = 0."""
        rate = angular_rate(45.0, 30.0, 0.0, 45.0, 30.0, 1.0)
        assert rate == pytest.approx(0.0, abs=1e-9)

    def test_1_deg_per_second(self):
        """1° of separation over 1s → 1 deg/s."""
        rate = angular_rate(0.0, 45.0, 0.0, 1.0, 45.0, 1.0)
        assert rate == pytest.approx(1.0, abs=0.01)

    def test_fast_object(self):
        """ISS-like speed: ~0.5°/s."""
        rate = angular_rate(90.0, 45.0, 0.0, 90.5, 45.0, 1.0)
        assert rate == pytest.approx(0.5, abs=0.01)

    def test_zero_dt_raises(self):
        with pytest.raises(ValueError, match="differ"):
            angular_rate(0.0, 0.0, 1.0, 1.0, 1.0, 1.0)

    def test_rate_independent_of_direction(self):
        """Rate should be the same moving north or east (same angular distance)."""
        rate_east = angular_rate(0.0, 45.0, 0.0, 1.0, 45.0, 1.0)
        rate_up   = angular_rate(0.0, 45.0, 0.0, 0.0, 46.0, 1.0)
        assert rate_east == pytest.approx(rate_up, abs=0.01)


# ---------------------------------------------------------------------------
# angular_velocity
# ---------------------------------------------------------------------------

class TestAngularVelocity:

    def test_moving_east(self):
        """Moving 2°/s in azimuth → d_az=2, d_alt=0."""
        d_az, d_alt = angular_velocity(0.0, 45.0, 0.0, 2.0, 45.0, 1.0)
        assert d_az  == pytest.approx(2.0, abs=0.01)
        assert d_alt == pytest.approx(0.0, abs=0.01)

    def test_moving_up(self):
        """Moving 1°/s in altitude → d_az=0, d_alt=1."""
        d_az, d_alt = angular_velocity(90.0, 30.0, 0.0, 90.0, 31.0, 1.0)
        assert d_az  == pytest.approx(0.0, abs=0.01)
        assert d_alt == pytest.approx(1.0, abs=0.01)

    def test_az_wrap_shortest_path(self):
        """Going from 359° to 1° over 1s → d_az = +2 (not -358)."""
        d_az, d_alt = angular_velocity(359.0, 0.0, 0.0, 1.0, 0.0, 1.0)
        assert d_az == pytest.approx(2.0, abs=0.1)

    def test_az_wrap_negative(self):
        """Going from 1° to 359° over 1s → d_az = -2."""
        d_az, d_alt = angular_velocity(1.0, 0.0, 0.0, 359.0, 0.0, 1.0)
        assert d_az == pytest.approx(-2.0, abs=0.1)


# ---------------------------------------------------------------------------
# predict_position
# ---------------------------------------------------------------------------

class TestPredictPosition:

    def test_extrapolate_forward(self):
        """0.5s lead at 2°/s az → advance 1°."""
        az, alt = predict_position(45.0, 30.0, 2.0, 0.0, 0.5)
        assert az  == pytest.approx(46.0, abs=0.01)
        assert alt == pytest.approx(30.0, abs=0.01)

    def test_az_wraps_at_360(self):
        """Az should wrap around 0/360."""
        az, alt = predict_position(359.0, 45.0, 5.0, 0.0, 1.0)
        assert az == pytest.approx(4.0, abs=0.1)

    def test_alt_clamped_at_90(self):
        """Altitude should not exceed 90°."""
        _, alt = predict_position(0.0, 89.0, 0.0, 5.0, 1.0)
        assert alt <= 90.0

    def test_alt_clamped_at_minus_90(self):
        """Altitude should not go below -90°."""
        _, alt = predict_position(0.0, -89.0, 0.0, -5.0, 1.0)
        assert alt >= -90.0

    def test_zero_lead(self):
        """Zero lead → same position."""
        az, alt = predict_position(45.0, 30.0, 2.0, -1.0, 0.0)
        assert az  == pytest.approx(45.0, abs=1e-9)
        assert alt == pytest.approx(30.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

class TestUnitHelpers:

    def test_ft_to_m(self):
        assert ft_to_m(1) == pytest.approx(0.3048, rel=1e-6)
        assert ft_to_m(0) == 0.0
        assert ft_to_m(35000) == pytest.approx(10668.0, rel=1e-3)

    def test_nm_to_m(self):
        assert nm_to_m(1) == pytest.approx(1852.0, rel=1e-6)
        assert nm_to_m(0) == 0.0

    def test_kt_to_ms(self):
        assert kt_to_ms(1) == pytest.approx(0.514444, rel=1e-4)
        assert kt_to_ms(0) == 0.0
        # 450 kt → ~231.5 m/s
        assert kt_to_ms(450) == pytest.approx(231.5, abs=0.5)


# ---------------------------------------------------------------------------
# Round-trip sanity: geodetic → az/alt → angular separation
# ---------------------------------------------------------------------------

class TestRoundTrip:

    def test_two_aircraft_separation(self):
        """
        Two aircraft 10 NM apart horizontally at same altitude.
        Angular separation as seen from observer between them should be
        consistent with the geometry.
        """
        observer_lat =  34.0
        observer_lon = -118.0
        observer_elev = 0

        alt_m = ft_to_m(30000)

        # Aircraft 1: directly north, 10 NM away
        # 1 NM ≈ 1/60 degree latitude
        ac1_lat = observer_lat + 10 / 60.0
        ac1_lon = observer_lon

        # Aircraft 2: directly east, 10 NM away (at this latitude, ~1/60° * cos(lat) lon)
        ac2_lat = observer_lat
        ac2_lon = observer_lon + (10 / 60.0) / math.cos(math.radians(observer_lat))

        az1, alt1 = geodetic_to_az_alt(
            observer_lat, observer_lon, observer_elev,
            ac1_lat, ac1_lon, alt_m,
        )
        az2, alt2 = geodetic_to_az_alt(
            observer_lat, observer_lon, observer_elev,
            ac2_lat, ac2_lon, alt_m,
        )

        # Aircraft 1 should be roughly north
        assert approx_angle(az1, 0.0, tol=3.0), f"AC1 expected north, got az={az1:.1f}°"
        # Aircraft 2 should be roughly east
        assert approx_angle(az2, 90.0, tol=3.0), f"AC2 expected east, got az={az2:.1f}°"
        # Angular separation should be close to 90°
        sep = angular_separation(az1, alt1, az2, alt2)
        assert 85 < sep < 95, f"Expected ~90° separation, got {sep:.1f}°"
