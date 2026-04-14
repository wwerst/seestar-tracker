"""
Coordinate transform utilities for seestar-tracker.

All angles in degrees unless otherwise stated.  Internally we use radians
for the spherical-trig heavy lifting, but every public function speaks degrees.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Geodetic → observer-centric azimuth / altitude
# ---------------------------------------------------------------------------

_WGS84_A = 6_378_137.0          # semi-major axis, metres
_WGS84_B = 6_356_752.314_245    # semi-minor axis, metres
_WGS84_E2 = 1 - (_WGS84_B / _WGS84_A) ** 2   # first eccentricity squared


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> tuple[float, float, float]:
    """Convert geodetic (WGS-84) coordinates to ECEF (Earth-Centred, Earth-Fixed)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    N = _WGS84_A / math.sqrt(1 - _WGS84_E2 * math.sin(lat) ** 2)
    x = (N + alt_m) * math.cos(lat) * math.cos(lon)
    y = (N + alt_m) * math.cos(lat) * math.sin(lon)
    z = (N * (1 - _WGS84_E2) + alt_m) * math.sin(lat)
    return x, y, z


def _ecef_to_enu(
    dx: float, dy: float, dz: float,
    ref_lat_deg: float, ref_lon_deg: float,
) -> tuple[float, float, float]:
    """Rotate an ECEF displacement vector into East-North-Up at the reference point."""
    lat = math.radians(ref_lat_deg)
    lon = math.radians(ref_lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    east  = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up    =  cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return east, north, up


def geodetic_to_az_alt(
    observer_lat: float,
    observer_lon: float,
    observer_elev_m: float,
    target_lat: float,
    target_lon: float,
    target_alt_m: float,
) -> tuple[float, float]:
    """
    Convert a target's geodetic position to observer-centric (azimuth, altitude).

    Parameters
    ----------
    observer_lat, observer_lon : degrees, WGS-84
    observer_elev_m            : observer elevation above WGS-84 ellipsoid, metres
    target_lat, target_lon     : degrees, WGS-84
    target_alt_m               : target altitude above WGS-84 ellipsoid, metres
                                 (for aircraft this is geometric altitude; use
                                 ft_to_m() to convert from feet)

    Returns
    -------
    (azimuth, altitude) in degrees
        azimuth  : 0 = North, 90 = East, 180 = South, 270 = West
        altitude : elevation angle above horizon; negative below horizon
    """
    ox, oy, oz = _geodetic_to_ecef(observer_lat, observer_lon, observer_elev_m)
    tx, ty, tz = _geodetic_to_ecef(target_lat, target_lon, target_alt_m)

    east, north, up = _ecef_to_enu(
        tx - ox, ty - oy, tz - oz,
        observer_lat, observer_lon,
    )

    slant = math.sqrt(east ** 2 + north ** 2 + up ** 2)
    if slant == 0:
        return 0.0, 90.0   # target is at observer

    alt_rad = math.asin(up / slant)
    az_rad  = math.atan2(east, north) % (2 * math.pi)

    return math.degrees(az_rad), math.degrees(alt_rad)


# ---------------------------------------------------------------------------
# Angular rate and separation
# ---------------------------------------------------------------------------

def angular_separation(az1: float, alt1: float, az2: float, alt2: float) -> float:
    """
    Great-circle angular separation between two points on the sky dome.

    Uses the spherical law of cosines (numerically stable for small angles via
    the haversine formula).

    Returns separation in degrees.
    """
    az1_r  = math.radians(az1)
    alt1_r = math.radians(alt1)
    az2_r  = math.radians(az2)
    alt2_r = math.radians(alt2)

    # Haversine — treat (az, alt) as (lon, lat) on the celestial sphere
    d_az  = az2_r - az1_r
    d_alt = alt2_r - alt1_r
    a = (
        math.sin(d_alt / 2) ** 2
        + math.cos(alt1_r) * math.cos(alt2_r) * math.sin(d_az / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(min(a, 1.0)))   # clamp for floating-point noise
    return math.degrees(c)


def angular_rate(
    az1: float, alt1: float, t1: float,
    az2: float, alt2: float, t2: float,
) -> float:
    """
    Compute angular rate in degrees/second between two sky positions.

    Parameters
    ----------
    az1, alt1 : azimuth and altitude at time t1 (degrees)
    t2, alt2  : azimuth and altitude at time t2 (degrees)
    t1, t2    : timestamps in seconds (e.g. time.monotonic())

    Returns
    -------
    Angular speed in degrees/second (always ≥ 0).
    Raises ValueError if t1 == t2.
    """
    dt = t2 - t1
    if dt == 0:
        raise ValueError("t1 and t2 must differ")
    sep = angular_separation(az1, alt1, az2, alt2)
    return sep / abs(dt)


# ---------------------------------------------------------------------------
# Angular velocity vector — for predictive lead
# ---------------------------------------------------------------------------

def angular_velocity(
    az1: float, alt1: float, t1: float,
    az2: float, alt2: float, t2: float,
) -> tuple[float, float]:
    """
    Return (d_az/dt, d_alt/dt) in degrees/second.

    az wraps at 0/360; we take the shortest path.
    """
    dt = t2 - t1
    if dt == 0:
        raise ValueError("t1 and t2 must differ")

    d_alt = alt2 - alt1

    # Shortest angular path for azimuth (handles 359→1 wrap)
    d_az = (az2 - az1 + 180) % 360 - 180

    return d_az / dt, d_alt / dt


def predict_position(
    az: float, alt: float,
    az_rate: float, alt_rate: float,
    lead_s: float,
) -> tuple[float, float]:
    """
    Extrapolate position by lead_s seconds at constant angular velocity.

    Returns (az, alt) in degrees with az normalised to [0, 360).
    """
    pred_az  = (az + az_rate * lead_s) % 360
    pred_alt = alt + alt_rate * lead_s
    pred_alt = max(-90.0, min(90.0, pred_alt))
    return pred_az, pred_alt


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def ft_to_m(feet: float) -> float:
    return feet * 0.3048


def nm_to_m(nautical_miles: float) -> float:
    return nautical_miles * 1_852.0


def kt_to_ms(knots: float) -> float:
    return knots * 0.514_444
