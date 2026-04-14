# seestar-tracker

Programmatic control of the ZWO Seestar S50 for tracking in-atmosphere and near-Earth targets (planes, ISS, satellites).

## Status

🚧 **Experimental** — this is a research project.

## Goal

Use the S50's TCP/JSON control protocol to implement real-time tracking of fast-moving targets:
- Planes departing/arriving LAX
- ISS passes
- Starlink trains
- Satellites

## Challenge

The S50 was designed for sidereal tracking (~0.004°/s). Fast-moving targets require much higher slew rates:

| Target | Angular Speed | S50 Feasible? |
|--------|--------------|---------------|
| Stars (sidereal) | 0.004°/s | ✅ Native |
| Moon | 0.004°/s + drift | ✅ Native |
| ISS overhead | ~0.5-1.0°/s | ⚠️ Probably too fast |
| Plane at 10,000ft, 5mi | ~2-5°/s | ❌ Almost certainly too fast |
| Plane at 30,000ft, 20mi | ~0.3-0.5°/s | ⚠️ Maybe possible |

The S50's slew motors can do GoTo slews at ~3-5°/s, but **continuous tracking** at those speeds
via the JSON API is uncharted territory. The motor update loop latency and command throughput
will determine what's actually achievable.

## Approach

1. **Reverse engineer the tracking protocol** — understand how the S50 handles GoTo vs continuous tracking commands
2. **Measure actual motor response** — latency, max sustained slew rate, command throughput
3. **ADS-B integration** — pull real-time plane positions from dump1090/FlightAware and compute angular rates
4. **Predictive tracking** — send position commands ahead of the target to compensate for latency
5. **ISS/satellite TLE tracking** — compute passes from TLE data, pre-compute slew path

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────┐
│ ADS-B / TLE │────▶│ Track Engine │────▶│ S50 (TCP) │
│   Source     │     │ (Python)     │     │ JSON API  │
└─────────────┘     └──────────────┘     └───────────┘
                          │
                     ┌────▼────┐
                     │ Web UI  │
                     │(optional)│
                     └─────────┘
```

## Dependencies

- Python 3.11+
- [seestar_alp](https://github.com/smart-underworld/seestar_alp) — reference for S50 protocol
- [pyModeS](https://github.com/junzis/pyModeS) — ADS-B decoding (for plane tracking)
- [skyfield](https://rhodesmill.org/skyfield/) — satellite/ISS pass computation
- RTL-SDR dongle (optional, for direct ADS-B reception)

## References

- [seestar_alp](https://github.com/smart-underworld/seestar_alp) — community S50 control project
- [seestar_run](https://github.com/smart-underworld/seestar_run) — lightweight S50 scripting
- ZWO official FAQ: "Seestar S50 does not support the tracking of near-Earth satellites" — challenge accepted.

## License

MIT
