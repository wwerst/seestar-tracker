"""
TLE-based satellite tracking using skyfield.

Provides:
  - SatelliteTracker   — compute az/alt, predict next pass, generate slew path
  - Pass               — dataclass describing a visible pass

TLE data is fetched from Celestrak on demand and cached for 6 hours.

Usage
-----
    import asyncio
    from datetime import datetime, timezone
    from seestar_tracker.satellite import SatelliteTracker

    async def main():
        tracker = SatelliteTracker(
            observer_lat=33.9425,
            observer_lon=-118.4081,
            observer_elev_m=38,
        )
        await tracker.load_tle("ISS (ZARYA)")
        now = datetime.now(timezone.utc)
        az, alt = tracker.get_position(now)
        print(f"ISS now: az={az:.1f}° alt={alt:.1f}°")

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import NamedTuple

import httpx
from skyfield.api import EarthSatellite, Loader, wgs84
from skyfield.api import load as skyfield_load
from skyfield.toposlib import GeographicPosition
from skyfield.units import Angle

log = logging.getLogger(__name__)

# Celestrak catalog URLs
CELESTRAK_BASE = "https://celestrak.org/GZIP/satcat-formatted.txt"
CELESTRAK_TLE  = "https://celestrak.org/SOCRATES/query.php"

# Named TLE groups on Celestrak
CELESTRAK_GROUPS: dict[str, str] = {
    "stations":   "https://celestrak.org/SOCRATES/query.php",
    "visual":     "https://celestrak.org/pub/TLE/catalog/visual.txt",
    "ISS":        "https://celestrak.org/pub/TLE/catalog/stations.txt",
    "starlink":   "https://celestrak.org/pub/TLE/catalog/starlink.txt",
    "active":     "https://celestrak.org/pub/TLE/catalog/active.txt",
}

CELESTRAK_STATIONS = "https://celestrak.org/pub/TLE/catalog/stations.txt"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class PassPoint(NamedTuple):
    """A single point on a satellite pass track."""
    timestamp: float    # Unix time
    az: float           # degrees
    alt: float          # degrees
    range_km: float     # slant range


@dataclass
class Pass:
    """Describes one visible satellite pass over the observer."""
    sat_name: str
    rise_time: datetime
    rise_az: float
    culmination_time: datetime
    culmination_alt: float
    set_time: datetime
    set_az: float
    max_alt: float      # same as culmination_alt
    duration_s: float   # seconds from rise to set
    track: list[PassPoint] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{self.sat_name}  rise {self.rise_time.strftime('%H:%M:%S')} az={self.rise_az:.0f}° "
            f"→ max {self.max_alt:.1f}° at {self.culmination_time.strftime('%H:%M:%S')} "
            f"→ set {self.set_time.strftime('%H:%M:%S')} az={self.set_az:.0f}°"
            f"  ({self.duration_s:.0f}s)"
        )


# ---------------------------------------------------------------------------
# TLE cache (module-level, avoids repeated HTTP fetches)
# ---------------------------------------------------------------------------

_tle_cache: dict[str, tuple[float, str, str]] = {}   # name → (fetched_at, line1, line2)
_TLE_TTL = 6 * 3600   # 6 hours


# ---------------------------------------------------------------------------
# SatelliteTracker
# ---------------------------------------------------------------------------

class SatelliteTracker:
    """
    Compute satellite positions and passes from TLE data.

    All positions are apparent topocentric az/alt, which is what you need
    to point a telescope.  Refraction is not applied — the S50 has no
    mechanism for atmospheric refraction correction at these speeds anyway.
    """

    def __init__(
        self,
        observer_lat: float,
        observer_lon: float,
        observer_elev_m: float = 0,
    ) -> None:
        """
        Parameters
        ----------
        observer_lat, observer_lon : WGS-84 degrees
        observer_elev_m            : metres above WGS-84 ellipsoid
        """
        self.observer_lat = observer_lat
        self.observer_lon = observer_lon
        self.observer_elev_m = observer_elev_m

        self._ts = skyfield_load.timescale()
        self._observer: GeographicPosition = wgs84.latlon(
            observer_lat, observer_lon, elevation_m=observer_elev_m
        )
        self._satellite: EarthSatellite | None = None
        self._sat_name: str = ""

    # ------------------------------------------------------------------
    # TLE loading
    # ------------------------------------------------------------------

    async def load_tle(
        self,
        name: str = "ISS (ZARYA)",
        url: str = CELESTRAK_STATIONS,
        force_refresh: bool = False,
    ) -> None:
        """
        Load TLE data for a named satellite.

        Parameters
        ----------
        name          : satellite name to search for in the TLE file
                        (case-insensitive prefix match)
        url           : Celestrak TLE URL; defaults to the stations catalog
        force_refresh : ignore cache and re-fetch
        """
        name_upper = name.strip().upper()
        cached = _tle_cache.get(name_upper)
        if cached and not force_refresh:
            fetched_at, line1, line2 = cached
            if time.time() - fetched_at < _TLE_TTL:
                log.debug("Using cached TLE for %s (age %.0fs)", name, time.time() - fetched_at)
                self._satellite = EarthSatellite(line1, line2, name, self._ts)
                self._sat_name = name
                return

        log.info("Fetching TLE for %s from %s …", name, url)
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            tle_text = resp.text

        line1, line2 = _find_tle_in_catalog(tle_text, name_upper)
        _tle_cache[name_upper] = (time.time(), line1, line2)
        self._satellite = EarthSatellite(line1, line2, name, self._ts)
        self._sat_name = name
        log.info("Loaded TLE for %s", name)

    def load_tle_direct(self, name: str, line1: str, line2: str) -> None:
        """Load TLE from already-fetched strings (useful for offline/testing)."""
        self._satellite = EarthSatellite(line1.strip(), line2.strip(), name, self._ts)
        self._sat_name = name

    def _require_tle(self) -> EarthSatellite:
        if self._satellite is None:
            raise RuntimeError("No TLE loaded — call load_tle() first")
        return self._satellite

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------

    def get_position(self, t: datetime | None = None) -> tuple[float, float]:
        """
        Return (azimuth, altitude) in degrees for the satellite at time t.

        t defaults to now (UTC).
        """
        sat = self._require_tle()
        if t is None:
            t = datetime.now(timezone.utc)

        sky_t = self._ts.from_datetime(t)
        difference = sat - self._observer
        topocentric = difference.at(sky_t)
        alt, az, distance = topocentric.altaz()

        return az.degrees, alt.degrees

    def get_position_with_range(self, t: datetime | None = None) -> tuple[float, float, float]:
        """Return (az_deg, alt_deg, range_km) for the satellite at time t."""
        sat = self._require_tle()
        if t is None:
            t = datetime.now(timezone.utc)

        sky_t = self._ts.from_datetime(t)
        difference = sat - self._observer
        topocentric = difference.at(sky_t)
        alt, az, distance = topocentric.altaz()

        return az.degrees, alt.degrees, distance.km

    # ------------------------------------------------------------------
    # Track generation (list of positions)
    # ------------------------------------------------------------------

    def get_track(
        self,
        start: datetime,
        duration_s: float,
        step_s: float = 1.0,
    ) -> list[PassPoint]:
        """
        Compute a slew path as a list of (timestamp, az, alt, range_km).

        Parameters
        ----------
        start      : UTC start time
        duration_s : total duration in seconds
        step_s     : time step between points in seconds

        Returns a list of PassPoint namedtuples.
        """
        sat = self._require_tle()
        track: list[PassPoint] = []
        n = int(duration_s / step_s) + 1
        start_ts = self._ts.from_datetime(start)

        for i in range(n):
            dt_days = (i * step_s) / 86_400.0
            sky_t = self._ts.tt_jd(start_ts.tt + dt_days)
            diff = (sat - self._observer).at(sky_t)
            alt, az, dist = diff.altaz()
            unix_t = start.timestamp() + i * step_s
            track.append(PassPoint(unix_t, az.degrees, alt.degrees, dist.km))

        return track

    # ------------------------------------------------------------------
    # Next pass prediction
    # ------------------------------------------------------------------

    def get_next_pass(
        self,
        after: datetime | None = None,
        min_alt: float = 10.0,
        search_days: float = 7.0,
        track_step_s: float = 1.0,
    ) -> Pass | None:
        """
        Find the next visible pass above min_alt degrees.

        Parameters
        ----------
        after       : search from this UTC time (default: now)
        min_alt     : minimum altitude in degrees to count as visible
        search_days : how many days ahead to search
        track_step_s: time step for the pass track

        Returns a Pass dataclass, or None if no pass found.
        """
        sat = self._require_tle()
        if after is None:
            after = datetime.now(timezone.utc)

        t0 = self._ts.from_datetime(after)
        t1 = self._ts.tt_jd(t0.tt + search_days)

        # skyfield find_events needs a horizon-angle threshold
        horizon_deg = min_alt
        times, events = sat.find_events(
            self._observer, t0, t1, altitude_degrees=horizon_deg
        )

        # events: 0 = rise, 1 = culmination, 2 = set
        if len(times) < 3:
            return None

        # Group into (rise, culmination, set) triplets
        passes: list[tuple] = []
        i = 0
        while i + 2 < len(times):
            if events[i] == 0 and events[i + 1] == 1 and events[i + 2] == 2:
                passes.append((times[i], times[i + 1], times[i + 2]))
                i += 3
            else:
                i += 1

        if not passes:
            return None

        rise_t, culm_t, set_t = passes[0]

        def _az_alt(t):
            diff = (sat - self._observer).at(t)
            alt, az, _ = diff.altaz()
            return az.degrees, alt.degrees

        rise_az, _   = _az_alt(rise_t)
        culm_az, culm_alt = _az_alt(culm_t)
        set_az, _    = _az_alt(set_t)

        rise_dt = rise_t.utc_datetime()
        culm_dt = culm_t.utc_datetime()
        set_dt  = set_t.utc_datetime()
        duration = (set_dt - rise_dt).total_seconds()

        track = self.get_track(rise_dt, duration, step_s=track_step_s)

        return Pass(
            sat_name=self._sat_name,
            rise_time=rise_dt,
            rise_az=rise_az,
            culmination_time=culm_dt,
            culmination_alt=culm_alt,
            set_time=set_dt,
            set_az=set_az,
            max_alt=culm_alt,
            duration_s=duration,
            track=track,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_tle_in_catalog(tle_text: str, name_upper: str) -> tuple[str, str]:
    """
    Search a multi-satellite TLE catalog for a named satellite.

    Standard 3-line TLE format:
        SATELLITE NAME
        1 NNNNNX YYYYDDD.DDDDDDDD  ...
        2 NNNNN  ...
    """
    lines = [ln.rstrip() for ln in tle_text.splitlines()]
    for i in range(len(lines) - 2):
        candidate = lines[i].strip().upper()
        # Prefix match — "ISS" matches "ISS (ZARYA)"
        if name_upper in candidate or candidate.startswith(name_upper):
            l1, l2 = lines[i + 1], lines[i + 2]
            if l1.startswith("1 ") and l2.startswith("2 "):
                return l1, l2
    raise ValueError(
        f"Satellite '{name_upper}' not found in TLE catalog.\n"
        f"Available names (first 20): "
        + ", ".join(
            lines[i].strip()
            for i in range(0, min(len(lines), 60), 3)
            if lines[i] and not lines[i].startswith(("1 ", "2 "))
        )
    )
