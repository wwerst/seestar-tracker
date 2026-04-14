"""
CLI entry points for seestar-tracker.

Commands
--------
  seestar-track status            — ping the S50 and show current position
  seestar-track plane             — track an aircraft by callsign or hex
  seestar-track iss               — track ISS (next pass or immediately)
  seestar-track sat               — track any TLE satellite
  seestar-track benchmark         — measure S50 motor response time

Run any command with --help for full options.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone

import click

# Shared S50 options applied to every command
_S50_OPTIONS = [
    click.option("--host", default="10.0.0.1", show_default=True, help="S50 IP address"),
    click.option("--port", default=4700, show_default=True, help="S50 TCP port"),
]


def add_s50_options(fn):
    for opt in reversed(_S50_OPTIONS):
        fn = opt(fn)
    return fn


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """seestar-tracker — real-time fast-target tracking for the ZWO Seestar S50."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@main.command()
@add_s50_options
@click.pass_context
def status(ctx: click.Context, host: str, port: int) -> None:
    """Ping the S50 and display current position and status."""

    async def _run() -> None:
        from .s50_client import S50Client
        async with S50Client(host, port) as client:
            click.echo(f"Connected to S50 at {host}:{port}")

            rtt = await client.ping()
            click.echo(f"Ping RTT: {rtt * 1000:.1f} ms")

            try:
                pos = await client.get_position()
                click.echo("Position:")
                for k, v in pos.items():
                    click.echo(f"  {k}: {v}")
            except Exception as exc:
                click.echo(f"  (could not get position: {exc})")

            try:
                stat = await client.get_status()
                click.echo("Status:")
                for k, v in stat.items():
                    click.echo(f"  {k}: {v}")
            except Exception as exc:
                click.echo(f"  (could not get status: {exc})")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# plane
# ---------------------------------------------------------------------------

@main.command()
@add_s50_options
@click.option("--callsign", default=None, help="Aircraft callsign (e.g. UAL123)")
@click.option("--hex", "hex_id", default=None, help="ICAO 24-bit hex code (e.g. a1b2c3)")
@click.option("--lat", required=True, type=float, help="Observer latitude (degrees)")
@click.option("--lon", required=True, type=float, help="Observer longitude (degrees)")
@click.option("--elev", default=0.0, type=float, help="Observer elevation (metres)")
@click.option("--duration", default=300, type=float, show_default=True, help="Tracking duration (seconds)")
@click.option("--rate", default=5.0, type=float, show_default=True, help="Update rate (Hz)")
@click.option(
    "--backend",
    default="auto",
    type=click.Choice(["auto", "dump1090", "opensky", "adsbx"]),
    show_default=True,
    help="ADS-B data source",
)
@click.option("--dump1090-url", default="http://localhost:8080/data/aircraft.json", show_default=True)
@click.option("--log-dir", default=None, help="Directory for CSV tracking logs")
@click.pass_context
def plane(
    ctx: click.Context,
    host: str, port: int,
    callsign: str | None, hex_id: str | None,
    lat: float, lon: float, elev: float,
    duration: float, rate: float,
    backend: str, dump1090_url: str,
    log_dir: str | None,
) -> None:
    """Track a specific aircraft by callsign or ICAO hex code."""
    if not callsign and not hex_id:
        click.echo("Error: provide --callsign or --hex", err=True)
        sys.exit(1)

    async def _run() -> None:
        from .s50_client import S50Client
        from .adsb_feed import ADSBFeed
        from .track_engine import TrackEngine

        feed = ADSBFeed(backend=backend, dump1090_url=dump1090_url)  # type: ignore[arg-type]
        async with S50Client(host, port) as client:
            engine = TrackEngine(
                client, lat, lon, elev,
                update_rate_hz=rate,
                log_dir=log_dir,
            )
            await engine.calibrate_latency()

            by = "callsign" if callsign else "hex"
            target = callsign or hex_id
            click.echo(f"Tracking {by}={target} for {duration:.0f}s at {rate} Hz")
            await engine.track_aircraft(feed, target, duration_s=duration, by=by)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# iss
# ---------------------------------------------------------------------------

@main.command()
@add_s50_options
@click.option("--lat", required=True, type=float, help="Observer latitude (degrees)")
@click.option("--lon", required=True, type=float, help="Observer longitude (degrees)")
@click.option("--elev", default=0.0, type=float, help="Observer elevation (metres)")
@click.option(
    "--next-pass",
    "next_pass",
    is_flag=True,
    default=False,
    help="Wait for the next pass and then track it",
)
@click.option("--duration", default=600, type=float, show_default=True, help="Max tracking duration (seconds)")
@click.option("--min-alt", default=10.0, type=float, show_default=True, help="Minimum altitude for pass prediction (degrees)")
@click.option("--log-dir", default=None, help="Directory for CSV tracking logs")
@click.pass_context
def iss(
    ctx: click.Context,
    host: str, port: int,
    lat: float, lon: float, elev: float,
    next_pass: bool, duration: float, min_alt: float,
    log_dir: str | None,
) -> None:
    """Track the ISS.  Use --next-pass to wait for and pre-compute the next visible pass."""

    async def _run() -> None:
        from .s50_client import S50Client
        from .satellite import SatelliteTracker
        from .track_engine import TrackEngine

        tracker = SatelliteTracker(lat, lon, elev)
        click.echo("Fetching latest ISS TLE from Celestrak …")
        await tracker.load_tle("ISS (ZARYA)")

        if next_pass:
            click.echo("Searching for next visible pass …")
            p = tracker.get_next_pass(min_alt=min_alt)
            if p is None:
                click.echo("No visible pass found in the next 7 days.")
                return

            click.echo(str(p))
            now = datetime.now(timezone.utc)
            wait_s = (p.rise_time - now).total_seconds()
            if wait_s > 0:
                click.echo(f"Waiting {wait_s:.0f}s until pass begins …")
                await asyncio.sleep(wait_s)

            async with S50Client(host, port) as client:
                engine = TrackEngine(client, lat, lon, elev, log_dir=log_dir)
                click.echo("Playing back pre-computed pass track …")
                await engine.track_precomputed(p.track)
        else:
            # Track in real-time right now
            async with S50Client(host, port) as client:
                engine = TrackEngine(client, lat, lon, elev, log_dir=log_dir)
                await engine.calibrate_latency()
                click.echo(f"Tracking ISS in real-time for {duration:.0f}s …")
                await engine.track_satellite(tracker, duration_s=duration)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# sat (generic satellite)
# ---------------------------------------------------------------------------

@main.command()
@add_s50_options
@click.argument("sat_name")
@click.option("--lat", required=True, type=float, help="Observer latitude (degrees)")
@click.option("--lon", required=True, type=float, help="Observer longitude (degrees)")
@click.option("--elev", default=0.0, type=float, help="Observer elevation (metres)")
@click.option("--duration", default=300, type=float, show_default=True)
@click.option("--url", default=None, help="Celestrak TLE URL (overrides built-in)")
@click.option("--log-dir", default=None)
@click.pass_context
def sat(
    ctx: click.Context,
    sat_name: str,
    host: str, port: int,
    lat: float, lon: float, elev: float,
    duration: float, url: str | None,
    log_dir: str | None,
) -> None:
    """Track any satellite by name (e.g. 'STARLINK-1234')."""

    async def _run() -> None:
        from .s50_client import S50Client
        from .satellite import SatelliteTracker, CELESTRAK_STATIONS

        tracker = SatelliteTracker(lat, lon, elev)
        tle_url = url or CELESTRAK_STATIONS
        click.echo(f"Fetching TLE for '{sat_name}' …")
        await tracker.load_tle(sat_name, url=tle_url)

        async with S50Client(host, port) as client:
            from .track_engine import TrackEngine
            engine = TrackEngine(client, lat, lon, elev, log_dir=log_dir)
            await engine.calibrate_latency()
            click.echo(f"Tracking {sat_name} for {duration:.0f}s …")
            await engine.track_satellite(tracker, duration_s=duration)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

@main.command()
@add_s50_options
@click.option("--samples", default=20, show_default=True, help="Number of command round-trips to measure")
@click.option(
    "--slew-test",
    "slew_test",
    is_flag=True,
    default=False,
    help="Also benchmark slew settle time with a small GoTo move",
)
@click.pass_context
def benchmark(
    ctx: click.Context,
    host: str, port: int,
    samples: int,
    slew_test: bool,
) -> None:
    """Measure S50 command latency, throughput, and (optionally) GoTo settle time."""

    async def _run() -> None:
        import statistics
        from .s50_client import S50Client

        async with S50Client(host, port) as client:
            click.echo(f"Connected to {host}:{port}")

            # --- Latency ---
            click.echo(f"\nMeasuring command round-trip latency ({samples} samples) …")
            rtts = []
            for i in range(samples):
                rtt = await client.ping()
                rtts.append(rtt * 1000)
                click.echo(f"  [{i + 1:3d}] {rtt * 1000:.1f} ms")
                await asyncio.sleep(0.05)

            click.echo(f"\nLatency statistics ({samples} pings):")
            click.echo(f"  min  = {min(rtts):.1f} ms")
            click.echo(f"  max  = {max(rtts):.1f} ms")
            click.echo(f"  mean = {statistics.mean(rtts):.1f} ms")
            click.echo(f"  σ    = {statistics.stdev(rtts):.1f} ms")
            click.echo(f"  p95  = {sorted(rtts)[int(0.95 * len(rtts))]:.1f} ms")

            # --- GoTo benchmark ---
            if slew_test:
                click.echo("\nGoTo settle-time benchmark …")
                click.echo("  Getting current position …")
                pos = await client.get_position()
                cur_az  = float(pos.get("az",  0))
                cur_alt = float(pos.get("alt", 45))

                targets = [
                    (cur_az + 1.0, cur_alt),
                    (cur_az - 1.0, cur_alt),
                    (cur_az, cur_alt + 0.5),
                    (cur_az, cur_alt - 0.5),
                ]

                import time
                settle_times = []
                for az, alt in targets:
                    az = az % 360
                    alt = max(5.0, min(89.0, alt))
                    t0 = time.monotonic()
                    await client.goto_az_alt(az, alt)
                    # Poll position until we're within 0.1° or 5s timeout
                    settle = None
                    for _ in range(50):
                        await asyncio.sleep(0.1)
                        try:
                            p = await client.get_position()
                            got_az  = float(p.get("az",  az))
                            got_alt = float(p.get("alt", alt))
                        except Exception:
                            continue
                        err = ((got_az - az) ** 2 + (got_alt - alt) ** 2) ** 0.5
                        if err < 0.1:
                            settle = time.monotonic() - t0
                            break
                    settle_times.append(settle)
                    click.echo(
                        f"  GoTo ({az:.2f}°, {alt:.2f}°) → "
                        + (f"settled in {settle * 1000:.0f} ms" if settle else "timeout (>5s)")
                    )

                valid = [s for s in settle_times if s is not None]
                if valid:
                    click.echo(f"\n  Mean settle time: {statistics.mean(valid) * 1000:.0f} ms")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
