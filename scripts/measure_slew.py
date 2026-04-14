#!/usr/bin/env python3
"""
Benchmark the S50's motor response time and command throughput.

This is the critical first script to run — the results tell you what's
physically achievable before you write a single line of tracking logic.

Measurements
------------
1. Command round-trip latency (N ping samples)
   → tells you the minimum lead time for predictive tracking

2. GoTo settling time
   → time from "goto_az_alt sent" until the mount reaches the target
   → done by polling get_position() in a tight loop

3. Maximum sustained GoTo rate
   → how many goto commands/second can the S50 accept before the queue overflows

4. Continuous slew speed characterisation
   → if the S50 supports scope_speed_slew, measure actual deg/s

Usage
-----
    python scripts/measure_slew.py --host 10.0.0.1 [--all] [--settle] [--throughput] [--speed]

Results are saved to measure_slew_<timestamp>.json for analysis.
"""

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seestar_tracker.s50_client import S50Client


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark S50 motor response time and command throughput.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host",    default="10.0.0.1", help="S50 IP address")
    p.add_argument("--port",    default=4700, type=int, help="S50 TCP port")
    p.add_argument("--samples", default=30,   type=int, help="Ping samples")
    p.add_argument(
        "--settle-move",
        default=1.0, type=float,
        help="GoTo move size in degrees for settle test",
    )
    p.add_argument(
        "--throughput-hz",
        nargs="+", type=float, default=[2, 5, 10, 20],
        help="Command rates (Hz) to test for throughput measurement",
    )
    p.add_argument(
        "--throughput-duration",
        default=5.0, type=float,
        help="Duration (seconds) per throughput rate test",
    )
    p.add_argument("--all",        action="store_true", help="Run all tests")
    p.add_argument("--latency",    action="store_true", help="Run latency test")
    p.add_argument("--settle",     action="store_true", help="Run GoTo settle test")
    p.add_argument("--throughput", action="store_true", help="Run command throughput test")
    p.add_argument("--output-dir", default=".", help="Directory to write JSON results")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Individual benchmark routines
# ---------------------------------------------------------------------------

async def bench_latency(client: S50Client, samples: int) -> dict:
    """Measure command round-trip latency."""
    print(f"\n{'─' * 55}")
    print(f"  LATENCY TEST  ({samples} samples)")
    print(f"{'─' * 55}")

    rtts_ms = []
    for i in range(samples):
        rtt = await client.ping()
        rtts_ms.append(rtt * 1000)
        print(f"  [{i + 1:3d}/{samples}]  {rtt * 1000:6.1f} ms")
        await asyncio.sleep(0.05)

    result = {
        "test": "latency",
        "samples": samples,
        "min_ms":    min(rtts_ms),
        "max_ms":    max(rtts_ms),
        "mean_ms":   statistics.mean(rtts_ms),
        "median_ms": statistics.median(rtts_ms),
        "stdev_ms":  statistics.stdev(rtts_ms) if len(rtts_ms) > 1 else 0,
        "p95_ms":    sorted(rtts_ms)[int(0.95 * len(rtts_ms))],
        "raw_ms":    rtts_ms,
    }

    print(f"\n  min={result['min_ms']:.1f}  max={result['max_ms']:.1f}  "
          f"mean={result['mean_ms']:.1f}  p95={result['p95_ms']:.1f} ms")
    print(f"  Recommended lead time: {result['p95_ms']:.0f} ms (p95)")
    return result


async def bench_settle(client: S50Client, move_deg: float) -> dict:
    """Measure GoTo settling time for small moves."""
    print(f"\n{'─' * 55}")
    print(f"  GOTO SETTLE TEST  (±{move_deg:.1f}° moves)")
    print(f"{'─' * 55}")

    # Get starting position
    pos = await client.get_position()
    start_az  = float(pos.get("az",  90.0))
    start_alt = float(pos.get("alt", 45.0))
    print(f"  Starting position: az={start_az:.2f}° alt={start_alt:.2f}°")

    # Build a set of small test moves
    moves: list[tuple[float, float]] = [
        ((start_az + move_deg) % 360, start_alt),
        ((start_az - move_deg) % 360, start_alt),
        (start_az, min(89.0, start_alt + move_deg)),
        (start_az, max(5.0,  start_alt - move_deg)),
    ]

    settle_results = []
    POLL_INTERVAL = 0.05   # 50 ms position poll
    TIMEOUT = 10.0          # seconds
    THRESHOLD_DEG = 0.05    # consider settled within this

    for target_az, target_alt in moves:
        print(f"\n  → GoTo az={target_az:.2f}° alt={target_alt:.2f}°")
        t_cmd = time.monotonic()
        await client.goto_az_alt(target_az, target_alt)
        t_sent = time.monotonic()

        poll_count = 0
        settled_ms = None
        final_az, final_alt = None, None

        while time.monotonic() - t_cmd < TIMEOUT:
            await asyncio.sleep(POLL_INTERVAL)
            poll_count += 1
            try:
                p = await client.get_position()
                got_az  = float(p.get("az",  target_az))
                got_alt = float(p.get("alt", target_alt))
                err_az  = abs((got_az  - target_az  + 180) % 360 - 180)
                err_alt = abs(got_alt - target_alt)
                err     = max(err_az, err_alt)
                elapsed_ms = (time.monotonic() - t_cmd) * 1000
                print(f"    t={elapsed_ms:6.0f}ms  pos=({got_az:.3f}°, {got_alt:.3f}°)  err={err:.3f}°")

                if err < THRESHOLD_DEG:
                    settled_ms = elapsed_ms
                    final_az, final_alt = got_az, got_alt
                    break
            except Exception as exc:
                print(f"    poll error: {exc}")

        if settled_ms is not None:
            print(f"  ✓ Settled in {settled_ms:.0f} ms ({poll_count} polls)")
        else:
            print(f"  ✗ Did not settle within {TIMEOUT:.0f}s")

        settle_results.append({
            "target_az":   target_az,
            "target_alt":  target_alt,
            "cmd_ms":      (t_sent - t_cmd) * 1000,
            "settle_ms":   settled_ms,
            "final_az":    final_az,
            "final_alt":   final_alt,
            "timed_out":   settled_ms is None,
        })

    valid = [r["settle_ms"] for r in settle_results if r["settle_ms"] is not None]
    summary = {
        "test": "settle",
        "move_deg": move_deg,
        "moves":    settle_results,
        "mean_settle_ms": statistics.mean(valid) if valid else None,
        "max_settle_ms":  max(valid) if valid else None,
        "timeout_count":  sum(1 for r in settle_results if r["timed_out"]),
    }

    if valid:
        print(f"\n  Mean settle: {summary['mean_settle_ms']:.0f} ms  Max: {summary['max_settle_ms']:.0f} ms")
    return summary


async def bench_throughput(
    client: S50Client,
    hz_list: list[float],
    duration_s: float,
) -> dict:
    """
    Measure maximum sustained command rate.

    We send goto commands at increasing Hz and check whether the S50 keeps
    up (responses arrive) or falls behind (responses stop or queue fills).
    """
    print(f"\n{'─' * 55}")
    print(f"  THROUGHPUT TEST  (rates: {hz_list} Hz, {duration_s:.0f}s each)")
    print(f"{'─' * 55}")

    pos = await client.get_position()
    base_az  = float(pos.get("az",  90.0))
    base_alt = float(pos.get("alt", 45.0))

    rate_results = []

    for target_hz in hz_list:
        interval = 1.0 / target_hz
        n_target = int(duration_s * target_hz)
        print(f"\n  Rate: {target_hz} Hz ({interval*1000:.0f} ms/cmd, {n_target} cmds)")

        sent = 0
        succeeded = 0
        overruns = 0
        t_start = time.monotonic()

        # Oscillate the telescope slightly so we send real (non-degenerate) GoTo commands
        for i in range(n_target):
            tick_start = time.monotonic()
            # Small oscillation: ±0.2° in azimuth
            target_az = base_az + 0.2 * (1 if i % 2 == 0 else -1)

            try:
                await client.goto_az_alt(target_az % 360, base_alt)
                succeeded += 1
            except Exception as exc:
                print(f"    cmd {i}: error — {exc}")
            sent += 1

            elapsed = time.monotonic() - tick_start
            wait = interval - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            else:
                overruns += 1

        total_elapsed = time.monotonic() - t_start
        actual_hz = sent / total_elapsed

        result = {
            "target_hz":    target_hz,
            "actual_hz":    actual_hz,
            "sent":         sent,
            "succeeded":    succeeded,
            "overruns":     overruns,
            "elapsed_s":    total_elapsed,
            "success_rate": succeeded / sent if sent else 0,
        }
        rate_results.append(result)
        print(
            f"  sent={sent}  ok={succeeded}  overruns={overruns}  "
            f"actual={actual_hz:.1f} Hz  success={result['success_rate']*100:.0f}%"
        )
        await asyncio.sleep(1.0)   # let the S50 settle between rates

    return {"test": "throughput", "results": rate_results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.WARNING,
    )
    log = logging.getLogger("measure_slew")

    run_all      = args.all or not (args.latency or args.settle or args.throughput)
    run_latency  = run_all or args.latency
    run_settle   = run_all or args.settle
    run_throughput = run_all or args.throughput

    print(f"\nS50 Motor Benchmark  —  {args.host}:{args.port}")
    print("=" * 55)

    results: dict = {
        "timestamp": datetime.now().isoformat(),
        "host": args.host,
        "port": args.port,
    }

    try:
        async with S50Client(args.host, args.port) as client:
            print("Connected.")
            status = await client.get_status()
            results["s50_status"] = status
            print(f"S50 status: {status}")

            if run_latency:
                results["latency"] = await bench_latency(client, args.samples)

            if run_settle:
                results["settle"] = await bench_settle(client, args.settle_move)

            if run_throughput:
                results["throughput"] = await bench_throughput(
                    client, args.throughput_hz, args.throughput_duration
                )

    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
    except ConnectionRefusedError:
        print(f"\nERROR: Cannot connect to S50 at {args.host}:{args.port}")
        print("Is the S50 powered on and in WiFi AP mode?")
        return 1

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"measure_slew_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 55}")
    print(f"Results saved to: {out_path}")
    print()

    # Print high-level summary
    if "latency" in results:
        lat = results["latency"]
        print(f"Latency:     mean={lat['mean_ms']:.1f} ms  p95={lat['p95_ms']:.1f} ms")
    if "settle" in results:
        s = results["settle"]
        if s.get("mean_settle_ms"):
            print(f"Settle time: mean={s['mean_settle_ms']:.0f} ms  max={s['max_settle_ms']:.0f} ms")
    if "throughput" in results:
        for r in results["throughput"]["results"]:
            print(
                f"Throughput @ {r['target_hz']} Hz:  "
                f"actual={r['actual_hz']:.1f} Hz  success={r['success_rate']*100:.0f}%"
            )

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
