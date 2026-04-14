"""
ADS-B aircraft position feed.

Supports three backends (tried in order if the primary fails):
  1. Local dump1090 / tar1090   — http://localhost:8080/data/aircraft.json
  2. OpenSky Network REST API   — https://opensky-network.org/api/states/all
  3. ADS-B Exchange (ADSBx)     — https://adsbexchange.com/api/aircraft/json/

Backend selection
-----------------
Pass ``backend="dump1090"`` / ``"opensky"`` / ``"adsbx"`` to the constructor,
or leave as ``"auto"`` to try them in order.

OpenSky and ADSBx return global snapshots; we filter by bounding box to keep
data volumes manageable.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Literal

import httpx

log = logging.getLogger(__name__)

Backend = Literal["auto", "dump1090", "opensky", "adsbx"]


@dataclass
class Aircraft:
    hex_id: str
    callsign: str | None
    lat: float
    lon: float
    alt_ft: float           # barometric altitude in feet
    track: float            # heading in degrees true (0-360)
    ground_speed_kt: float  # ground speed in knots
    vertical_rate: float    # ft/min; positive = climbing
    timestamp: float        # Unix time of this position fix
    squawk: str | None = None
    on_ground: bool = False
    category: str | None = None   # e.g. "A3" (ICAO category)

    @property
    def alt_m(self) -> float:
        return self.alt_ft * 0.3048


class ADSBFeed:
    """Pull real-time aircraft positions from an ADS-B source."""

    DUMP1090_URL    = "http://localhost:8080/data/aircraft.json"
    OPENSKY_URL     = "https://opensky-network.org/api/states/all"
    ADSBX_URL       = "https://adsbexchange.com/api/aircraft/json/"

    # OpenSky column indices in the state vector
    _OSN_ICAO24 = 0
    _OSN_CALLSIGN = 1
    _OSN_LONGITUDE = 5
    _OSN_LATITUDE = 6
    _OSN_BARO_ALT = 7
    _OSN_ON_GROUND = 8
    _OSN_VELOCITY = 9
    _OSN_TRUE_TRACK = 10
    _OSN_VERTICAL_RATE = 11
    _OSN_GEO_ALT = 13
    _OSN_SQUAWK = 14
    _OSN_TIME_POS = 3

    def __init__(
        self,
        backend: Backend = "auto",
        dump1090_url: str = DUMP1090_URL,
        opensky_bbox: tuple[float, float, float, float] | None = None,
        refresh_interval: float = 3.0,
        http_timeout: float = 5.0,
    ) -> None:
        """
        Parameters
        ----------
        backend           : data source selection
        dump1090_url      : override dump1090 endpoint (e.g. remote Pi)
        opensky_bbox      : (lat_min, lat_max, lon_min, lon_max) filter for OpenSky;
                            defaults to None (global, slow)
        refresh_interval  : seconds between automatic refreshes when using
                            start_polling() / stop_polling()
        http_timeout      : per-request HTTP timeout in seconds
        """
        self.backend = backend
        self.dump1090_url = dump1090_url
        self.opensky_bbox = opensky_bbox
        self.refresh_interval = refresh_interval

        self._client = httpx.AsyncClient(timeout=http_timeout, follow_redirects=True)
        self._cache: list[Aircraft] = []
        self._cache_ts: float = 0.0
        self._poll_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await self.stop_polling()
        await self._client.aclose()

    async def __aenter__(self) -> "ADSBFeed":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def start_polling(self) -> None:
        """Start a background task that refreshes the aircraft list periodically."""
        if self._poll_task and not self._poll_task.done():
            return
        self._poll_task = asyncio.create_task(self._poll_loop(), name="adsb-poll")

    async def stop_polling(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

    async def _poll_loop(self) -> None:
        while True:
            try:
                self._cache = await self._fetch_all()
                self._cache_ts = time.monotonic()
            except Exception as exc:
                log.warning("ADS-B refresh failed: %s", exc)
            await asyncio.sleep(self.refresh_interval)

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    async def get_aircraft(self, max_age: float = 30.0) -> list[Aircraft]:
        """Return all known aircraft; refresh if cache is stale."""
        if time.monotonic() - self._cache_ts > max_age:
            self._cache = await self._fetch_all()
            self._cache_ts = time.monotonic()
        return self._cache

    async def get_aircraft_by_callsign(self, callsign: str) -> Aircraft | None:
        """Find aircraft by callsign (case-insensitive, strip whitespace)."""
        target = callsign.strip().upper()
        for ac in await self.get_aircraft():
            if ac.callsign and ac.callsign.strip().upper() == target:
                return ac
        return None

    async def get_aircraft_by_hex(self, hex_id: str) -> Aircraft | None:
        """Find aircraft by ICAO 24-bit hex code (e.g. 'a1b2c3')."""
        target = hex_id.strip().lower()
        for ac in await self.get_aircraft():
            if ac.hex_id.lower() == target:
                return ac
        return None

    async def get_nearby(
        self,
        lat: float,
        lon: float,
        radius_nm: float,
    ) -> list[Aircraft]:
        """Return aircraft within radius_nm nautical miles of (lat, lon)."""
        radius_m = radius_nm * 1_852.0
        result = []
        for ac in await self.get_aircraft():
            if _haversine_m(lat, lon, ac.lat, ac.lon) <= radius_m:
                result.append(ac)
        return result

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    async def _fetch_all(self) -> list[Aircraft]:
        backends = (
            [self.backend] if self.backend != "auto"
            else ["dump1090", "opensky"]
        )
        last_exc: Exception | None = None
        for b in backends:
            try:
                if b == "dump1090":
                    return await self._fetch_dump1090()
                elif b == "opensky":
                    return await self._fetch_opensky()
                elif b == "adsbx":
                    return await self._fetch_adsbx()
            except Exception as exc:
                log.warning("Backend %s failed: %s", b, exc)
                last_exc = exc
        raise RuntimeError(f"All ADS-B backends failed; last error: {last_exc}") from last_exc

    # ---- dump1090 -------------------------------------------------------

    async def _fetch_dump1090(self) -> list[Aircraft]:
        resp = await self._client.get(self.dump1090_url)
        resp.raise_for_status()
        data = resp.json()
        aircraft_list: list[dict] = data.get("aircraft", data if isinstance(data, list) else [])
        return [self._parse_dump1090(a) for a in aircraft_list if self._valid_position(a)]

    @staticmethod
    def _valid_position(a: dict) -> bool:
        return (
            a.get("lat") is not None
            and a.get("lon") is not None
            and a.get("alt_baro") is not None
        )

    @staticmethod
    def _parse_dump1090(a: dict) -> Aircraft:
        return Aircraft(
            hex_id=a.get("hex", "").strip(),
            callsign=a.get("flight", "").strip() or None,
            lat=float(a["lat"]),
            lon=float(a["lon"]),
            alt_ft=float(a.get("alt_baro", a.get("alt_geom", 0))),
            track=float(a.get("track", 0)),
            ground_speed_kt=float(a.get("gs", 0)),
            vertical_rate=float(a.get("baro_rate", a.get("geom_rate", 0))),
            timestamp=float(a.get("seen_pos", a.get("seen", 0))),
            squawk=a.get("squawk"),
            on_ground=bool(a.get("on_ground", False)),
            category=a.get("category"),
        )

    # ---- OpenSky --------------------------------------------------------

    async def _fetch_opensky(self) -> list[Aircraft]:
        params: dict = {}
        if self.opensky_bbox:
            lat_min, lat_max, lon_min, lon_max = self.opensky_bbox
            params = {
                "lamin": lat_min, "lamax": lat_max,
                "lomin": lon_min, "lomax": lon_max,
            }
        resp = await self._client.get(self.OPENSKY_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        states: list[list] = data.get("states") or []
        now = time.time()
        result = []
        for sv in states:
            if sv[self._OSN_LATITUDE] is None or sv[self._OSN_LONGITUDE] is None:
                continue
            alt_baro = sv[self._OSN_BARO_ALT]
            alt_geo  = sv[self._OSN_GEO_ALT]
            alt_ft   = ((alt_baro or alt_geo or 0) / 0.3048)
            result.append(Aircraft(
                hex_id=sv[self._OSN_ICAO24] or "",
                callsign=(sv[self._OSN_CALLSIGN] or "").strip() or None,
                lat=float(sv[self._OSN_LATITUDE]),
                lon=float(sv[self._OSN_LONGITUDE]),
                alt_ft=alt_ft,
                track=float(sv[self._OSN_TRUE_TRACK] or 0),
                ground_speed_kt=float((sv[self._OSN_VELOCITY] or 0) / 0.514_444),
                vertical_rate=float((sv[self._OSN_VERTICAL_RATE] or 0) * 196.85),
                timestamp=float(sv[self._OSN_TIME_POS] or now),
                squawk=sv[self._OSN_SQUAWK],
                on_ground=bool(sv[self._OSN_ON_GROUND]),
            ))
        return result

    # ---- ADS-B Exchange -------------------------------------------------

    async def _fetch_adsbx(self) -> list[Aircraft]:
        resp = await self._client.get(self.ADSBX_URL)
        resp.raise_for_status()
        data = resp.json()
        ac_list: list[dict] = data.get("ac", [])
        result = []
        for a in ac_list:
            try:
                if a.get("lat") is None or a.get("lon") is None:
                    continue
                result.append(Aircraft(
                    hex_id=a.get("icao", a.get("hex", "")).strip(),
                    callsign=a.get("call", a.get("flight", "")).strip() or None,
                    lat=float(a["lat"]),
                    lon=float(a["lon"]),
                    alt_ft=float(a.get("alt_baro", a.get("alt", 0))),
                    track=float(a.get("track", a.get("trk", 0))),
                    ground_speed_kt=float(a.get("gs", 0)),
                    vertical_rate=float(a.get("baro_rate", 0)),
                    timestamp=float(a.get("seen_pos", 0)),
                    squawk=a.get("squawk"),
                    on_ground=a.get("alt_baro", "").lower() == "ground" if isinstance(a.get("alt_baro"), str) else False,
                    category=a.get("category"),
                ))
            except (KeyError, ValueError, TypeError) as exc:
                log.debug("Skipping malformed ADSBx record: %s", exc)
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two geodetic points."""
    R = 6_371_000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + (
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))
