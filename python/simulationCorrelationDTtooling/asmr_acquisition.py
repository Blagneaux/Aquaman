#!/usr/bin/env python3
"""Headless UDP pressure and optional CNC telemetry acquisition for ASMR 2.0."""

from __future__ import annotations

import bisect
import csv
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


LogCallback = Callable[[str], None]


@dataclass(frozen=True)
class AcquisitionStats:
    packets_received: int
    valid_samples: int
    rows_written: int
    complete_rows: int
    partial_rows: int
    held_slots: int
    blank_slots: int
    malformed_packets: int
    ignored_devices: int


class CncTelemetryState:
    def __init__(self, history_limit: int = 20_000):
        self._lock = threading.Lock()
        self._history_limit = int(history_limit)
        self._timestamps: list[int] = []
        self._samples: list[tuple[float, float, float, float, int]] = []
        self._latest = (0.0, 0.0, 0.0, 0.0, 0)
        self._valid = False
        self._last_rx_monotonic = 0.0

    def update(
        self,
        timestamp_us: int,
        x: float,
        y: float,
        z: float,
        fourth: float,
        moving: int,
    ) -> None:
        sample = (float(x), float(y), float(z), float(fourth), 1 if moving else 0)
        with self._lock:
            self._latest = sample
            self._valid = True
            self._last_rx_monotonic = time.monotonic()
            timestamp_us = int(timestamp_us)
            if timestamp_us <= 0:
                return
            if not self._timestamps or timestamp_us >= self._timestamps[-1]:
                self._timestamps.append(timestamp_us)
                self._samples.append(sample)
            else:
                index = bisect.bisect_left(self._timestamps, timestamp_us)
                self._timestamps.insert(index, timestamp_us)
                self._samples.insert(index, sample)
            overflow = len(self._timestamps) - self._history_limit
            if overflow > 0:
                del self._timestamps[:overflow]
                del self._samples[:overflow]

    def latest(self) -> tuple[float, float, float, float, int, bool]:
        with self._lock:
            return (*self._latest, self._valid)

    def nearest(
        self,
        timestamp_us: int,
        max_delta_us: int,
    ) -> tuple[float, float, float, float, int, bool, int | None]:
        with self._lock:
            if not self._timestamps:
                return 0.0, 0.0, 0.0, 0.0, 0, False, None
            index = bisect.bisect_left(self._timestamps, int(timestamp_us))
            candidates = []
            if index < len(self._timestamps):
                candidates.append(index)
            if index > 0:
                candidates.append(index - 1)
            best = min(candidates, key=lambda item: abs(self._timestamps[item] - timestamp_us))
            delta = int(self._timestamps[best] - timestamp_us)
            if abs(delta) > int(max_delta_us):
                return 0.0, 0.0, 0.0, 0.0, 0, False, delta
            return (*self._samples[best], True, delta)

    def mark_idle_if_stale(self, timeout_s: float = 3.0) -> None:
        with self._lock:
            if (
                self._valid
                and self._latest[4] == 1
                and self._last_rx_monotonic
                and time.monotonic() - self._last_rx_monotonic > timeout_s
            ):
                self._latest = (*self._latest[:4], 0)


class CncTelemetryReceiver(threading.Thread):
    SAMPLE_STRUCT = struct.Struct("<QffffB3x")

    def __init__(
        self,
        state: CncTelemetryState,
        bind_ip: str,
        port: int,
        source_ip: str | None,
        stop_event: threading.Event,
        log: LogCallback,
    ):
        super().__init__(daemon=True, name="asmr-cnc-telemetry")
        self.state = state
        self.bind_ip = bind_ip
        self.port = int(port)
        self.source_ip = source_ip or None
        self.stop_event = stop_event
        self.log = log
        self.ready_event = threading.Event()
        self.error: Exception | None = None
        self.sock: socket.socket | None = None

    def run(self) -> None:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.bind_ip, self.port))
            self.sock.settimeout(0.5)
            self.ready_event.set()
            self.log(
                f"[CNC telemetry] Listening on {self.bind_ip}:{self.port}"
                + (f", source={self.source_ip}" if self.source_ip else "")
            )
            while not self.stop_event.is_set():
                self.state.mark_idle_if_stale()
                try:
                    data, address = self.sock.recvfrom(8192)
                except socket.timeout:
                    continue
                if self.source_ip and address[0] != self.source_ip:
                    continue
                if not data or len(data) % self.SAMPLE_STRUCT.size:
                    self.log(
                        f"[CNC telemetry] Ignored malformed packet of {len(data)} bytes "
                        f"from {address[0]}"
                    )
                    continue
                for sample in self.SAMPLE_STRUCT.iter_unpack(data):
                    self.state.update(*sample)
        except Exception as exc:  # noqa: BLE001 - surfaced through session startup.
            self.error = exc
            self.ready_event.set()
            self.log(f"[CNC telemetry] Receiver stopped with error: {exc}")
        finally:
            if self.sock:
                self.sock.close()


class UDPPressureReceiver(threading.Thread):
    SAMPLE_STRUCT = struct.Struct("=Q8h")

    def __init__(
        self,
        bind_ip: str,
        port: int,
        sample_period_us: int,
        expected_devices: int,
        output_path: Path,
        telemetry: CncTelemetryState | None,
        telemetry_align_us: int,
        telemetry_fallback_latest: bool,
        partial_epoch_timeout_s: float,
        socket_buffer_bytes: int,
        stop_event: threading.Event,
        log: LogCallback,
    ):
        super().__init__(daemon=True, name="asmr-pressure-udp")
        self.bind_ip = bind_ip
        self.port = int(port)
        self.sample_period_us = int(sample_period_us)
        self.expected_devices = max(1, int(expected_devices))
        self.output_path = Path(output_path)
        self.telemetry = telemetry
        self.telemetry_align_us = int(telemetry_align_us)
        self.telemetry_fallback_latest = bool(telemetry_fallback_latest)
        self.partial_epoch_timeout_s = float(partial_epoch_timeout_s)
        self.socket_buffer_bytes = int(socket_buffer_bytes)
        self.stop_event = stop_event
        self.log = log

        self.ready_event = threading.Event()
        self.error: Exception | None = None
        self.sock: socket.socket | None = None
        self._csv_file = None
        self._writer = None
        self._epochs: dict[int, dict[str, object]] = {}
        self._device_slots: dict[str, int] = {}
        self._last_values: list[tuple[int, ...] | None] = [None] * self.expected_devices
        self._last_finalized_epoch = -1
        self._last_flush = time.monotonic()

        self.packets_received = 0
        self.valid_samples = 0
        self.rows_written = 0
        self.complete_rows = 0
        self.partial_rows = 0
        self.held_slots = 0
        self.blank_slots = 0
        self.malformed_packets = 0
        self.ignored_devices = 0
        self._ignored_ips: set[str] = set()

    @property
    def stats(self) -> AcquisitionStats:
        return AcquisitionStats(
            packets_received=self.packets_received,
            valid_samples=self.valid_samples,
            rows_written=self.rows_written,
            complete_rows=self.complete_rows,
            partial_rows=self.partial_rows,
            held_slots=self.held_slots,
            blank_slots=self.blank_slots,
            malformed_packets=self.malformed_packets,
            ignored_devices=self.ignored_devices,
        )

    def _device_slot(self, source_ip: str) -> int | None:
        if source_ip in self._device_slots:
            return self._device_slots[source_ip]
        if len(self._device_slots) >= self.expected_devices:
            if source_ip not in self._ignored_ips:
                self._ignored_ips.add(source_ip)
                self.ignored_devices += 1
                self.log(f"[UDP] Ignoring additional sensor source {source_ip}")
            return None
        slot = len(self._device_slots)
        self._device_slots[source_ip] = slot
        self.log(f"[UDP] Assigned {source_ip} to ESP{slot + 1}")
        return slot

    def _write_header(self) -> None:
        header = ["timestamp"]
        for device in range(self.expected_devices):
            header.extend(f"ESP{device + 1}_Sensor{sensor + 1}" for sensor in range(8))
        header.extend(["CNC_X", "CNC_Y", "CNC_Z", "CNC_A", "CNC_Moving"])
        self._writer.writerow(header)

    def _telemetry_columns(self, timestamp_us: int) -> list[object]:
        if self.telemetry is None:
            return ["", "", "", "", 0]
        x, y, z, fourth, moving, valid, _ = self.telemetry.nearest(
            timestamp_us, self.telemetry_align_us
        )
        if not valid and self.telemetry_fallback_latest:
            x, y, z, fourth, moving, valid = self.telemetry.latest()
        if not valid:
            return ["", "", "", "", 0]
        return [
            f"{x:.3f}".replace(".", ","),
            f"{y:.3f}".replace(".", ","),
            f"{z:.3f}".replace(".", ","),
            f"{fourth:.3f}".replace(".", ","),
            int(moving),
        ]

    def _write_epoch(self, epoch: int, state: dict[str, object], complete: bool) -> None:
        slots = state["slots"]
        timestamp = int(epoch + self.sample_period_us // 2)
        row: list[object] = [timestamp]
        for slot in range(self.expected_devices):
            values = slots.get(slot)
            if values is None:
                values = self._last_values[slot]
                if values is None:
                    row.extend([""] * 8)
                    self.blank_slots += 1
                    continue
                self.held_slots += 1
            else:
                self._last_values[slot] = tuple(values)
            row.extend(values)
        row.extend(self._telemetry_columns(timestamp))
        self._writer.writerow(row)
        self.rows_written += 1
        if complete:
            self.complete_rows += 1
        else:
            self.partial_rows += 1
        self._last_finalized_epoch = max(self._last_finalized_epoch, epoch)

    def _expire_epochs(self, force: bool = False) -> None:
        now = time.monotonic()
        while self._epochs:
            epoch = min(self._epochs)
            state = self._epochs[epoch]
            slots = state["slots"]
            complete = len(slots) >= self.expected_devices
            stale = force or now - float(state["created_at"]) >= self.partial_epoch_timeout_s
            if not complete and not stale:
                break
            self._epochs.pop(epoch)
            self._write_epoch(epoch, state, complete)

    def run(self) -> None:
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.socket_buffer_bytes)
            self.sock.bind((self.bind_ip, self.port))
            self.sock.settimeout(0.2)
            self._csv_file = self.output_path.open("w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._csv_file, delimiter=";")
            self._write_header()
            self.ready_event.set()
            self.log(f"[UDP] Recording {self.bind_ip}:{self.port} to {self.output_path}")

            while not self.stop_event.is_set():
                try:
                    data, address = self.sock.recvfrom(65535)
                except socket.timeout:
                    self._expire_epochs()
                    if time.monotonic() - self._last_flush >= 1.0:
                        self._csv_file.flush()
                        self._last_flush = time.monotonic()
                    continue

                self.packets_received += 1
                if not data:
                    continue
                if len(data) % self.SAMPLE_STRUCT.size:
                    self.malformed_packets += 1
                    self.log(
                        f"[UDP] Ignored malformed packet of {len(data)} bytes "
                        f"from {address[0]}"
                    )
                    continue
                slot = self._device_slot(address[0])
                if slot is None:
                    continue

                for sample in self.SAMPLE_STRUCT.iter_unpack(data):
                    timestamp = int(sample[0])
                    if timestamp < 1_000_000:
                        continue
                    epoch = (timestamp // self.sample_period_us) * self.sample_period_us
                    if epoch <= self._last_finalized_epoch:
                        continue
                    state = self._epochs.setdefault(
                        epoch, {"slots": {}, "created_at": time.monotonic()}
                    )
                    state["slots"][slot] = tuple(sample[1:])
                    self.valid_samples += 1
                self._expire_epochs()
        except Exception as exc:  # noqa: BLE001 - recorded and surfaced by the session.
            self.error = exc
            self.ready_event.set()
            self.log(f"[UDP] Receiver stopped with error: {exc}")
        finally:
            if self._writer:
                self._expire_epochs(force=True)
            if self._csv_file:
                self._csv_file.flush()
                self._csv_file.close()
            if self.sock:
                self.sock.close()


class AcquisitionSession:
    def __init__(
        self,
        sensor_config: dict,
        telemetry_config: dict,
        output_path: Path,
        log: LogCallback = print,
    ):
        self.sensor_config = sensor_config
        self.telemetry_config = telemetry_config
        self.output_path = output_path
        self.log = log
        self.stop_event = threading.Event()
        self.telemetry_state: CncTelemetryState | None = None
        self.telemetry_receiver: CncTelemetryReceiver | None = None
        self.pressure_receiver: UDPPressureReceiver | None = None

    def start(self, timeout_s: float = 5.0) -> None:
        if bool(self.telemetry_config.get("enabled", True)):
            self.telemetry_state = CncTelemetryState(
                history_limit=int(self.telemetry_config.get("history_samples", 20_000))
            )
            self.telemetry_receiver = CncTelemetryReceiver(
                state=self.telemetry_state,
                bind_ip=str(self.telemetry_config.get("bind_ip", "0.0.0.0")),
                port=int(self.telemetry_config.get("port", 12345)),
                source_ip=self.telemetry_config.get("source_ip") or None,
                stop_event=self.stop_event,
                log=self.log,
            )
            self.telemetry_receiver.start()
            self.telemetry_receiver.ready_event.wait(timeout_s)
            if self.telemetry_receiver.error:
                # Telemetry is optional: log the failure and continue pressure acquisition.
                self.log(
                    f"[CNC telemetry] Continuing without telemetry: "
                    f"{self.telemetry_receiver.error}"
                )
                self.telemetry_state = None

        self.pressure_receiver = UDPPressureReceiver(
            bind_ip=str(self.sensor_config.get("bind_ip", "0.0.0.0")),
            port=int(self.sensor_config.get("port", 12346)),
            sample_period_us=int(self.sensor_config.get("sample_period_us", 5000)),
            expected_devices=int(self.sensor_config.get("expected_devices", 1)),
            output_path=self.output_path,
            telemetry=self.telemetry_state,
            telemetry_align_us=int(self.telemetry_config.get("align_delta_us", 100_000)),
            telemetry_fallback_latest=bool(
                self.telemetry_config.get("fallback_latest", True)
            ),
            partial_epoch_timeout_s=float(
                self.sensor_config.get("partial_epoch_timeout_s", 0.75)
            ),
            socket_buffer_bytes=int(
                self.sensor_config.get("socket_buffer_bytes", 8 * 1024 * 1024)
            ),
            stop_event=self.stop_event,
            log=self.log,
        )
        self.pressure_receiver.start()
        if not self.pressure_receiver.ready_event.wait(timeout_s):
            self.stop()
            raise TimeoutError("Timed out waiting for the UDP pressure receiver to become ready.")
        if self.pressure_receiver.error:
            error = self.pressure_receiver.error
            self.stop()
            raise RuntimeError(f"UDP pressure receiver failed to start: {error}") from error

    def stop(self, join_timeout_s: float = 3.0) -> AcquisitionStats:
        self.stop_event.set()
        if self.pressure_receiver:
            self.pressure_receiver.join(join_timeout_s)
        if self.telemetry_receiver:
            self.telemetry_receiver.join(join_timeout_s)
        if self.pressure_receiver is None:
            return AcquisitionStats(0, 0, 0, 0, 0, 0, 0, 0, 0)
        if self.pressure_receiver.is_alive():
            raise TimeoutError("UDP pressure receiver did not stop cleanly.")
        if self.pressure_receiver.error:
            raise RuntimeError(
                f"UDP pressure receiver failed: {self.pressure_receiver.error}"
            ) from self.pressure_receiver.error
        return self.pressure_receiver.stats

    def __enter__(self) -> "AcquisitionSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()
