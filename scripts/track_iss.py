#!/usr/bin/env python3
"""
Track the ISS during its next visible pass.

Usage
-----
    # Predict and wait for the next pass, then track it:
    python scripts/track_iss.py --lat 34.01 --lon -118.50 --next-pass

    # Track the ISS right now (if it's above the horizon):
    python scripts/track_iss.py --lat 34.01 --lon -118.50

    # Show next pass without connecting to the S50:
    python scripts/track_iss.py --lat 34.01 --lon -118.50 --predict-only

Workflow
--------
1. Fetch the latest ISS TLE from Celestrak.
2. Compute the next visible pass for your location.
3. (If --next-pass) Wait until the pass begins.
4. Play back the pre-computed slew path to the S50 at 1-second intervals.

Note: the ISS moves at ~0.5-1.0°/s at peak altitude during a pass.  The S50
can slew at ~3-5°/s for GoTo moves, so tracking *may* be possible for passes
that don't go directly overhead.  Passes with max altitude < 45° are the best
candidates.
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seestar_tracker.satellite import SatelliteTracker
from seestar_tracker.s50_client import S50Client
from seestar_tracker.track_engine import TrackEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Track the ISS with the ZWO Seestar S50.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lat",   required=True, type=float, help="Observer latitude (degrees N)")
    p.add_argument("--lon",   required=True, type=float, help="Observer longitude (degrees E)")
    p.add_argument("--elev",  default=0.0,   type=float, help="Observer elevation (metres)")

    p.add_argument("--host", default="10.0.0.1", help="S50 IP address")
    p.add_argument("--port", default=4700, type=int,    help="S50 TCP port")

    p.add_argument(
        "--next-pass",
        action="store_true",
        help="Predict the next visible pass and wait for it",
    )
    p.add_argument(
        "--predict-only",
        action="store_true",
        help="Print next pass info without connecting to the S50",
    )
    p.add_argument(
        "--min-alt", default=10.0, type=float,
        help="Minimum altitude (degrees) to consider a pass visible",
    )
    p.add_argument("--duration", default=600, type=float, help="Max real-time tracking duration (s)")
    p.add_argument("--log-dir",  default="logs", help="Directory for CSV tracking logs")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p.parse_args()


async def main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    log = logging.getLogger("track_iss")

    tracker = SatelliteTracker(args.lat, args.lon, args.elev)

    log.info("Fetching latest ISS TLE from Celestrak …")
    await tracker.load_tle("ISS (ZARYA)")

    # --- Predict-only mode ---
    if args.predict_only or args.next_pass:
        log.info("Computing next pass (min_alt=%.0f°) …", args.min_alt)
        p = tracker.get_next_pass(min_alt=args.min_alt)
        if p is None:
            log.error("No visible pass found in the next 7 days.")
            return 1

        print()
        print("=" * 60)
        print(f"  Next ISS pass for lat={args.lat:.3f} lon={args.lon:.3f}")
        print("=" * 60)
        print(f"  Rise:        {p.rise_time.strftime('%Y-%m-%d %H:%M:%S UTC')}  az={p.rise_az:.0f}°")
        print(f"  Max Alt:     {p.max_alt:.1f}°  at {p.culmination_time.strftime('%H:%M:%S UTC')}")
        print(f"  Set:         {p.set_time.strftime('%H:%M:%S UTC')}  az={p.set_az:.0f}°")
        print(f"  Duration:    {p.duration_s:.0f}s")
        print(f"  Track pts:   {len(p.track)}")

        # Angular rate stats
        if len(p.track) >= 2:
            from seestar_tracker.coord_utils import angular_rate
            rates = []
            for i in range(1, len(p.track)):
                pt_prev = p.track[i - 1]
                pt_curr = p.track[i]
                rate = angular_rate(
                    pt_prev.az, pt_prev.alt, pt_prev.timestamp,
                    pt_curr.az,  pt_curr.alt,  pt_curr.timestamp,
                )
                rates.append(rate)
            print(f"  Angular rate: min={min(rates):.2f}°/s  max={max(rates):.2f}°/s  mean={sum(rates)/len(rates):.2f}°/s")
            fast = sum(1 for r in rates if r > 1.0)
            print(f"  Seconds >1°/s (likely too fast for S50): {fast}/{len(rates)}")
        print()

        if args.predict_only:
            return 0

    # --- Connect and track ---
    if args.next_pass:
        # p was computed above
        now = datetime.now(timezone.utc)
        wait_s = (p.rise_time - now).total_seconds()
        if wait_s > 300:
            log.info("Pass begins in %.0f minutes — waiting …", wait_s / 60)
        elif wait_s > 0:
            log.info("Pass begins in %.0f seconds — waiting …", wait_s)
        else:
            log.info("Pass already in progress (%.0fs past rise)", -wait_s)

        if wait_s > 0:
            await asyncio.sleep(wait_s)

        async with S50Client(args.host, args.port) as client:
            engine = TrackEngine(client, args.lat, args.lon, args.elev, log_dir=args.log_dir)
            log.info("Playing back pre-computed ISS track (%d points) …", len(p.track))
            await engine.track_precomputed(p.track)

    else:
        # Real-time tracking right now
        az, alt = tracker.get_position()
        log.info("ISS current position: az=%.1f° alt=%.1f°", az, alt)
        if alt < -5:
            log.warning("ISS is currently well below the horizon (alt=%.1f°).  Use --next-pass.", alt)

        async with S50Client(args.host, args.port) as client:
            engine = TrackEngine(client, args.lat, args.lon, args.elev, log_dir=args.log_dir)
            await engine.calibrate_latency()
            log.info("Tracking ISS in real-time for %.0fs …", args.duration)
            await engine.track_satellite(tracker, duration_s=args.duration)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main(parse_args())))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
