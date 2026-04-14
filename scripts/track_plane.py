#!/usr/bin/env python3
"""
Track a specific aircraft by callsign or ICAO hex code.

Usage
-----
    python scripts/track_plane.py --callsign UAL123 --lat 34.01 --lon -118.50
    python scripts/track_plane.py --hex a1b2c3   --lat 34.01 --lon -118.50 --backend dump1090

The script:
  1. Connects to the S50 and calibrates command latency.
  2. Connects to the ADS-B feed and locates the target aircraft.
  3. Runs the tracking loop until the aircraft is out of range or the
     duration expires.

ADS-B backend options:
  auto     — try dump1090 (local RTL-SDR), then OpenSky Network
  dump1090 — local dump1090/tar1090 at http://localhost:8080/data/aircraft.json
  opensky  — OpenSky Network REST API (anonymous, rate-limited)
  adsbx    — ADS-B Exchange API
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Make the src tree importable when running directly from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seestar_tracker.s50_client import S50Client
from seestar_tracker.adsb_feed import ADSBFeed
from seestar_tracker.track_engine import TrackEngine, TrackingAborted


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Track an aircraft with the ZWO Seestar S50.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--callsign", help="Aircraft callsign, e.g. UAL123")
    target.add_argument("--hex", dest="hex_id", help="ICAO hex code, e.g. a1b2c3")

    p.add_argument("--lat",   required=True, type=float, help="Observer latitude (degrees N)")
    p.add_argument("--lon",   required=True, type=float, help="Observer longitude (degrees E)")
    p.add_argument("--elev",  default=0.0,   type=float, help="Observer elevation (metres)")

    p.add_argument("--host",     default="10.0.0.1",  help="S50 IP address")
    p.add_argument("--port",     default=4700,          type=int, help="S50 TCP port")
    p.add_argument("--duration", default=300,           type=float, help="Max tracking duration (seconds)")
    p.add_argument("--rate",     default=5.0,           type=float, help="Command rate (Hz)")

    p.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "dump1090", "opensky", "adsbx"],
        help="ADS-B data backend",
    )
    p.add_argument(
        "--dump1090-url",
        default="http://localhost:8080/data/aircraft.json",
        help="dump1090 aircraft.json URL",
    )
    p.add_argument("--log-dir", default="logs", help="Directory for CSV tracking logs")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p.parse_args()


async def main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    log = logging.getLogger("track_plane")

    by = "callsign" if args.callsign else "hex"
    target = args.callsign or args.hex_id
    log.info("Target: %s=%s", by, target)
    log.info("Observer: lat=%.4f lon=%.4f elev=%.0fm", args.lat, args.lon, args.elev)

    feed = ADSBFeed(
        backend=args.backend,           # type: ignore[arg-type]
        dump1090_url=args.dump1090_url,
    )

    # Quick sanity check — is the aircraft visible right now?
    log.info("Checking ADS-B feed for %s …", target)
    if by == "callsign":
        ac = await feed.get_aircraft_by_callsign(target)
    else:
        ac = await feed.get_aircraft_by_hex(target)

    if ac is None:
        log.error("Aircraft %s=%s not found in ADS-B feed.  Is it airborne?", by, target)
        return 1

    log.info(
        "Found: %s  lat=%.3f lon=%.3f alt=%.0fft  gs=%.0fkt  track=%.0f°",
        ac.callsign or ac.hex_id, ac.lat, ac.lon, ac.alt_ft, ac.ground_speed_kt, ac.track,
    )

    try:
        async with S50Client(args.host, args.port) as client:
            log.info("Connected to S50 at %s:%d", args.host, args.port)
            engine = TrackEngine(
                client,
                args.lat, args.lon, args.elev,
                update_rate_hz=args.rate,
                log_dir=args.log_dir,
            )
            await engine.calibrate_latency()
            log.info(
                "Starting tracking loop for %.0fs at %.1f Hz (lead=%.0fms)",
                args.duration, args.rate, engine.lead_s * 1000,
            )
            await engine.track_aircraft(feed, target, duration_s=args.duration, by=by)

    except TrackingAborted as exc:
        log.error("Tracking aborted: %s", exc)
        return 2
    except KeyboardInterrupt:
        log.info("Interrupted by user.")

    await feed.close()
    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
