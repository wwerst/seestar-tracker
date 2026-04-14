"""
Core tracking loop — drives the S50 to follow a moving target.

Design
------
The loop runs at a configurable rate (default 5 Hz) and for each tick:
  1. Queries the position source (ADS-B feed or satellite tracker).
  2. Converts to observer-centric az/alt.
  3. Computes current angular rate (deg/s) and applies a predictive lead
     to compensate for S50 command latency.
  4. Sends a goto_az_alt command to the S50.
  5. Logs the tick to CSV for post-analysis.

Rate limits
-----------
The S50's TCP socket can absorb roughly 10-20 JSON commands/s without overflow.
We conservatively default to 5 Hz (200 ms between commands).  Adjust with
``update_rate_hz``.

Predictive lead
---------------
We estimate the S50 command latency empirically on startup (via ping) and then
extrapolate the target position by that many seconds into the future.  This
dramatically reduces tracking lag for fast-moving targets.

Warning thresholds
------------------
If the target's angular rate exceeds ``warn_rate_deg_s`` the tracker logs a
warning.  If it exceeds ``abort_rate_deg_s`` the tracker stops to protect the
motors.
"""

from __future__ import annotations

import asyncio
import csv
import logging
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Callable, Coroutine

from .s50_client import S50Client
from .adsb_feed import ADSBFeed
from .satellite import SatelliteTracker, PassPoint
from .coord_utils import (
    geodetic_to_az_alt,
    angular_velocity,
    predict_position,
    angular_rate,
    ft_to_m,
)

log = logging.getLogger(__name__)

# Type alias: an async callable that returns (az_deg, alt_deg) or None
PositionSource = Callable[[], Coroutine[None, None, tuple[float, float] | None]]


class TrackingAborted(Exception):
    """Raised when tracking is stopped due to excessive angular rate."""


class TrackEngine:
    """
    Real-time tracking loop.

    Parameters
    ----------
    client          : connected S50Client
    observer_lat    : observer WGS-84 latitude in degrees
    observer_lon    : observer WGS-84 longitude in degrees
    observer_elev_m : observer elevation above ellipsoid in metres
    update_rate_hz  : command rate to the S50 (Hz)
    lead_s          : predictive-lead time in seconds (overrides auto-ping)
    warn_rate_deg_s : log a warning above this angular rate
    abort_rate_deg_s: stop tracking above this angular rate
    log_dir         : directory for CSV tracking logs (None = no logging)
    """

    def __init__(
        self,
        client: S50Client,
        observer_lat: float,
        observer_lon: float,
        observer_elev_m: float = 0,
        update_rate_hz: float = 5.0,
        lead_s: float | None = None,
        warn_rate_deg_s: float = 1.0,
        abort_rate_deg_s: float = 5.0,
        log_dir: Path | str | None = None,
    ) -> None:
        self.client = client
        self.observer_lat = observer_lat
        self.observer_lon = observer_lon
        self.observer_elev_m = observer_elev_m
        self.update_rate_hz = update_rate_hz
        self._lead_s = lead_s
        self.warn_rate_deg_s = warn_rate_deg_s
        self.abort_rate_deg_s = abort_rate_deg_s
        self.log_dir = Path(log_dir) if log_dir else None

        self._interval = 1.0 / update_rate_hz
        self._running = False
        self._recent_positions: deque[tuple[float, float, float]] = deque(maxlen=5)
        # (timestamp, az, alt) — used to compute angular rate

    # ------------------------------------------------------------------
    # Lead-time calibration
    # ------------------------------------------------------------------

    async def calibrate_latency(self, samples: int = 5) -> float:
        """Measure S50 round-trip latency and set predictive lead time."""
        log.info("Calibrating S50 command latency (%d samples) …", samples)
        rtts = []
        for _ in range(samples):
            rtt = await self.client.ping()
            rtts.append(rtt)
            await asyncio.sleep(0.05)
        avg = sum(rtts) / len(rtts)
        self._lead_s = avg
        log.info("Mean RTT = %.1f ms → using lead = %.1f ms", avg * 1000, avg * 1000)
        return avg

    @property
    def lead_s(self) -> float:
        return self._lead_s if self._lead_s is not None else 0.2   # 200 ms default

    # ------------------------------------------------------------------
    # Public tracking methods
    # ------------------------------------------------------------------

    async def track_aircraft(
        self,
        feed: ADSBFeed,
        target_id: str,
        duration_s: float = 300,
        by: str = "hex",   # "hex" or "callsign"
    ) -> None:
        """
        Track a specific aircraft from the ADS-B feed.

        Parameters
        ----------
        feed       : ADSBFeed instance (should already be polling or fresh)
        target_id  : ICAO hex code or callsign
        duration_s : maximum tracking duration
        by         : "hex" or "callsign"
        """
        log.info("Tracking aircraft %s=%s for %.0fs", by, target_id, duration_s)

        async def position_source() -> tuple[float, float] | None:
            if by == "callsign":
                ac = await feed.get_aircraft_by_callsign(target_id)
            else:
                ac = await feed.get_aircraft_by_hex(target_id)

            if ac is None:
                log.warning("Aircraft %s not found in feed", target_id)
                return None

            # Aircraft alt is barometric feet; convert to geometric metres
            # (close enough for angular-position purposes)
            target_alt_m = ft_to_m(ac.alt_ft)
            az, alt = geodetic_to_az_alt(
                self.observer_lat, self.observer_lon, self.observer_elev_m,
                ac.lat, ac.lon, target_alt_m,
            )
            return az, alt

        await self._run_loop(position_source, duration_s, label=f"aircraft:{target_id}")

    async def track_satellite(
        self,
        tracker: SatelliteTracker,
        duration_s: float = 300,
    ) -> None:
        """
        Track a satellite using pre-loaded TLE data.

        Parameters
        ----------
        tracker    : SatelliteTracker with TLE already loaded
        duration_s : maximum tracking duration
        """
        log.info("Tracking satellite %s for %.0fs", tracker._sat_name, duration_s)

        def position_source_sync() -> tuple[float, float] | None:
            az, alt = tracker.get_position()
            if alt < -5.0:
                log.warning("Satellite below horizon (alt=%.1f°)", alt)
                return None
            return az, alt

        async def position_source() -> tuple[float, float] | None:
            return position_source_sync()

        await self._run_loop(position_source, duration_s, label=f"satellite:{tracker._sat_name}")

    async def track_precomputed(
        self,
        track: list[PassPoint],
    ) -> None:
        """
        Execute a pre-computed slew path (e.g. from SatelliteTracker.get_track()).

        This variant is deterministic — it plays back the track at the scheduled
        timestamps regardless of how long each goto takes.
        """
        if not track:
            return

        log.info("Playing back pre-computed track (%d points)", len(track))
        log_file, writer = self._open_log(label="precomputed")
        try:
            for pt in track:
                now = time.time()
                wait = pt.timestamp - now
                if wait > 0:
                    await asyncio.sleep(wait)
                elif wait < -1.0:
                    log.warning("Track playback %.1fs behind schedule, skipping point", -wait)
                    continue

                az, alt = pt.az, pt.alt
                if alt < 0:
                    continue

                await self.client.goto_az_alt(az, alt)
                if writer:
                    writer.writerow({
                        "timestamp": pt.timestamp,
                        "cmd_az": az,
                        "cmd_alt": alt,
                        "range_km": pt.range_km,
                    })
        finally:
            if log_file:
                log_file.close()

    async def track_coordinates(
        self,
        coord_generator: AsyncIterator[tuple[float, float]],
        duration_s: float = 300,
    ) -> None:
        """
        Track using an async generator that yields (az, alt) pairs.

        Useful for custom position sources.
        """
        log.info("Tracking from custom coordinate generator for %.0fs", duration_s)

        async def position_source() -> tuple[float, float] | None:
            try:
                return await coord_generator.__anext__()
            except StopAsyncIteration:
                return None

        await self._run_loop(position_source, duration_s, label="custom")

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    async def _run_loop(
        self,
        source: PositionSource,
        duration_s: float,
        label: str = "track",
    ) -> None:
        self._running = True
        self._recent_positions.clear()

        deadline = time.monotonic() + duration_s
        log_file, writer = self._open_log(label)

        tick_count = 0
        missed_ticks = 0

        try:
            while self._running and time.monotonic() < deadline:
                tick_start = time.monotonic()

                # 1. Get raw target position
                result = await source()
                if result is None:
                    await asyncio.sleep(self._interval)
                    continue

                raw_az, raw_alt = result

                if raw_alt < 0:
                    log.debug("Target below horizon (alt=%.1f°), waiting …", raw_alt)
                    await asyncio.sleep(self._interval)
                    continue

                # 2. Record for rate computation
                now = time.monotonic()
                self._recent_positions.append((now, raw_az, raw_alt))

                # 3. Compute angular rate and apply predictive lead
                az, alt = raw_az, raw_alt
                rate = 0.0
                if len(self._recent_positions) >= 2:
                    t1, az1, alt1 = self._recent_positions[-2]
                    t2, az2, alt2 = self._recent_positions[-1]
                    try:
                        rate = angular_rate(az1, alt1, t1, az2, alt2, t2)
                        d_az, d_alt = angular_velocity(az1, alt1, t1, az2, alt2, t2)
                        az, alt = predict_position(raw_az, raw_alt, d_az, d_alt, self.lead_s)
                    except ValueError:
                        pass

                # 4. Rate-limit checks
                if rate > self.abort_rate_deg_s:
                    log.error(
                        "Angular rate %.2f°/s exceeds abort threshold %.2f°/s — stopping",
                        rate, self.abort_rate_deg_s,
                    )
                    await self.client.stop()
                    raise TrackingAborted(f"Angular rate {rate:.2f}°/s exceeded abort threshold")

                if rate > self.warn_rate_deg_s:
                    log.warning("High angular rate: %.2f°/s", rate)

                # 5. Send goto command
                send_t = time.monotonic()
                try:
                    await self.client.goto_az_alt(az, alt)
                except Exception as exc:
                    log.error("goto_az_alt failed: %s", exc)

                # 6. Log tick
                tick_count += 1
                if writer:
                    writer.writerow({
                        "timestamp": time.time(),
                        "raw_az": raw_az,
                        "raw_alt": raw_alt,
                        "cmd_az": az,
                        "cmd_alt": alt,
                        "ang_rate_deg_s": rate,
                        "lead_s": self.lead_s,
                        "send_ms": (time.monotonic() - send_t) * 1000,
                        "tick_ms": (time.monotonic() - tick_start) * 1000,
                    })

                # 7. Sleep remainder of tick interval
                elapsed = time.monotonic() - tick_start
                sleep_time = self._interval - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    missed_ticks += 1
                    log.debug("Tick overrun by %.1f ms", -sleep_time * 1000)

        finally:
            self._running = False
            if log_file:
                log_file.close()

            log.info(
                "Tracking complete: %d ticks, %d missed (%.1f%% overrun)",
                tick_count, missed_ticks,
                100 * missed_ticks / max(tick_count, 1),
            )

    def stop(self) -> None:
        """Signal the tracking loop to stop at the next tick."""
        self._running = False

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _open_log(self, label: str):
        """Open a CSV log file.  Returns (file, DictWriter) or (None, None)."""
        if not self.log_dir:
            return None, None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.log_dir / f"track_{label}_{ts}.csv"
        f = open(path, "w", newline="")
        fieldnames = [
            "timestamp", "raw_az", "raw_alt", "cmd_az", "cmd_alt",
            "ang_rate_deg_s", "lead_s", "send_ms", "tick_ms",
            "range_km",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        log.info("Logging track to %s", path)
        return f, writer
