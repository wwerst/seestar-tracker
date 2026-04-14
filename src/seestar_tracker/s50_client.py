"""
Async TCP/JSON client for the ZWO Seestar S50 smart telescope.

Protocol (reverse-engineered from seestar_alp):
  - TCP connection to S50's WiFi AP, default 10.0.0.1:4700
  - Newline-delimited JSON messages in both directions
  - Request:  {"id": <int>, "method": "<name>", "params": {...}}
  - Response: {"id": <int>, "result": {...}} or {"id": <int>, "error": {...}}
  - Unsolicited events also arrive on the same socket (id == 0 or absent)

Usage
-----
    async with S50Client() as client:
        await client.connect()
        pos = await client.get_position()
        await client.goto_az_alt(180.0, 45.0)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

log = logging.getLogger(__name__)


class S50Error(Exception):
    """Raised when the S50 returns an error response."""


class S50Client:
    """Async TCP client for the ZWO Seestar S50 smart telescope."""

    DEFAULT_HOST = "10.0.0.1"
    DEFAULT_PORT = 4700
    CONNECT_TIMEOUT = 5.0    # seconds
    RESPONSE_TIMEOUT = 10.0  # seconds
    RECONNECT_DELAY = 2.0    # seconds between reconnect attempts

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        auto_reconnect: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.auto_reconnect = auto_reconnect

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._read_task: asyncio.Task | None = None

        self._cmd_id = 0
        # Map from command id → Future that will hold the response
        self._pending: dict[int, asyncio.Future] = {}
        # Unsolicited events (id absent or 0) go here
        self._event_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=64)

        self._connected = False
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "S50Client":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self, host: str | None = None, port: int | None = None) -> None:
        """Open the TCP connection to the S50 and start the background reader."""
        if host:
            self.host = host
        if port:
            self.port = port

        log.info("Connecting to S50 at %s:%d …", self.host, self.port)
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port),
            timeout=self.CONNECT_TIMEOUT,
        )
        self._connected = True
        self._read_task = asyncio.create_task(
            self._read_loop(), name="s50-reader"
        )
        log.info("Connected to S50.")

    async def disconnect(self) -> None:
        """Close the TCP connection cleanly."""
        self._connected = False
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None
        log.info("Disconnected from S50.")

    async def _reconnect(self) -> None:
        log.warning("S50 connection lost, reconnecting in %.1fs …", self.RECONNECT_DELAY)
        await asyncio.sleep(self.RECONNECT_DELAY)
        try:
            await self.connect()
        except Exception as exc:
            log.error("Reconnect failed: %s", exc)

    # ------------------------------------------------------------------
    # Background reader
    # ------------------------------------------------------------------

    async def _read_loop(self) -> None:
        """Read newline-delimited JSON from the socket, dispatch to waiters."""
        assert self._reader is not None
        try:
            while self._connected:
                line = await self._reader.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode().strip())
                except json.JSONDecodeError as exc:
                    log.debug("Bad JSON from S50: %s — %r", exc, line[:120])
                    continue

                msg_id = msg.get("id", 0)
                if msg_id and msg_id in self._pending:
                    fut = self._pending.pop(msg_id)
                    if not fut.done():
                        if "error" in msg:
                            fut.set_exception(S50Error(msg["error"]))
                        else:
                            fut.set_result(msg.get("result", {}))
                else:
                    # Unsolicited event — drop oldest if queue is full
                    if self._event_queue.full():
                        try:
                            self._event_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    self._event_queue.put_nowait(msg)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("S50 read loop died: %s", exc)
            # Cancel pending futures so callers don't hang
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("S50 connection lost"))
            self._pending.clear()
            if self.auto_reconnect and self._connected:
                asyncio.create_task(self._reconnect())

    # ------------------------------------------------------------------
    # Low-level command
    # ------------------------------------------------------------------

    async def send_command(
        self,
        method: str,
        params: dict | None = None,
    ) -> dict:
        """
        Send a JSON command and await the matching response.

        Returns the ``result`` dict from the response.
        Raises ``S50Error`` on protocol-level errors.
        """
        async with self._lock:
            self._cmd_id += 1
            cmd_id = self._cmd_id

        payload: dict[str, Any] = {"id": cmd_id, "method": method}
        if params:
            payload["params"] = params

        line = json.dumps(payload) + "\n"
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[cmd_id] = fut

        assert self._writer is not None, "Not connected — call connect() first"
        self._writer.write(line.encode())
        await self._writer.drain()

        log.debug("→ %s", line.rstrip())
        result = await asyncio.wait_for(fut, timeout=self.RESPONSE_TIMEOUT)
        log.debug("← %s", result)
        return result

    # ------------------------------------------------------------------
    # Movement commands
    # ------------------------------------------------------------------

    async def goto_ra_dec(self, ra: float, dec: float) -> dict:
        """
        Slew to J2000 equatorial coordinates.

        Parameters
        ----------
        ra  : right ascension in decimal hours (0 – 24)
        dec : declination in degrees (-90 – +90)
        """
        return await self.send_command(
            "scope_goto",
            {"ra": round(ra, 6), "dec": round(dec, 6)},
        )

    async def goto_az_alt(self, az: float, alt: float) -> dict:
        """
        Slew to horizontal (azimuth/altitude) coordinates.

        Parameters
        ----------
        az  : azimuth in degrees, 0 = North, 90 = East
        alt : altitude in degrees above horizon
        """
        return await self.send_command(
            "scope_goto_azel",
            {"az": round(az, 4), "alt": round(alt, 4)},
        )

    async def slew(self, axis: str, speed: float) -> dict:
        """
        Begin continuous slew on one axis.

        Parameters
        ----------
        axis  : "ra" / "dec" or "az" / "alt"
        speed : speed in degrees/second (positive or negative)
        """
        return await self.send_command(
            "scope_speed_slew",
            {"axis": axis, "speed": round(speed, 4)},
        )

    async def stop(self) -> dict:
        """Stop all motion."""
        return await self.send_command("scope_stop_slew")

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    async def get_position(self) -> dict:
        """
        Return current telescope position.

        Typical response keys: ra, dec, az, alt, ra_j2000, dec_j2000
        """
        return await self.send_command("scope_get_equ_coord")

    async def get_status(self) -> dict:
        """Return general device status (tracking, connection, battery, etc.)."""
        return await self.send_command("get_device_state")

    async def ping(self) -> float:
        """
        Send a no-op and return round-trip time in seconds.

        Uses get_device_state as the ping command because the S50 doesn't
        have a dedicated echo/ping method.
        """
        t0 = time.monotonic()
        await self.get_status()
        return time.monotonic() - t0

    # ------------------------------------------------------------------
    # Camera commands
    # ------------------------------------------------------------------

    async def start_capture(self, exposure_ms: int = 10_000) -> dict:
        """Start image capture with the given exposure time in milliseconds."""
        return await self.send_command(
            "start_capture",
            {"exp_ms": exposure_ms},
        )

    async def stop_capture(self) -> dict:
        """Stop ongoing capture."""
        return await self.send_command("stop_capture")

    async def autofocus(self) -> dict:
        """Trigger the autofocus routine."""
        return await self.send_command("start_auto_focuse")

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    async def next_event(self, timeout: float = 5.0) -> dict | None:
        """
        Wait for the next unsolicited event from the S50.

        Returns None on timeout.
        """
        try:
            return await asyncio.wait_for(self._event_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
