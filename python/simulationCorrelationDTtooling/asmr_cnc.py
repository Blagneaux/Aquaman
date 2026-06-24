#!/usr/bin/env python3
"""Trajectory generation and direct GRBL-style serial control for ASMR 2.0."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping


LogCallback = Callable[[str], None]


@dataclass(frozen=True)
class MotionPoint:
    x_mm: float
    dt_s: float
    feed_mm_min: float


@dataclass(frozen=True)
class TowProgram:
    label: str
    re_value: float
    h_value: float
    x_start_mm: float
    x_end_mm: float
    y_mm: float
    z_mm: float
    fourth_mm: float
    preposition_lines: tuple[str, ...]
    tow_lines: tuple[str, ...]
    points: tuple[MotionPoint, ...]


def speed_mm_s(re_value: float, length_scale_m: float, viscosity_m2_s: float) -> float:
    if length_scale_m <= 0 or viscosity_m2_s <= 0:
        raise ValueError("length_scale_m and viscosity_m2_s must be positive.")
    return float(re_value) * float(viscosity_m2_s) / float(length_scale_m) * 1000.0


def tank_y_mm(h_value: float, tank: Mapping[str, object]) -> float:
    width_mm = float(tank["width_mm"])
    wall_width_mm = float(tank.get("wall_width_mm", 0.0))
    y0_mm = float(tank.get("y0_mm", 0.0))
    h_mm_per_unit = float(tank["h_mm_per_unit"])
    h_max = width_mm / 2.0 - wall_width_mm
    return -(y0_mm + h_max - float(h_value) * h_mm_per_unit)


def build_trapezoid_profile(
    re_value: float,
    tank: Mapping[str, object],
    fluid: Mapping[str, object],
) -> tuple[list[float], list[float]]:
    target_speed = speed_mm_s(
        re_value,
        length_scale_m=float(fluid.get("length_scale_m", 0.05)),
        viscosity_m2_s=float(fluid.get("kinematic_viscosity_m2_s", 1e-6)),
    )
    if target_speed <= 0:
        raise ValueError("Computed towing speed must be positive.")

    tank_length = float(tank["length_mm"])
    travel_margin = float(tank.get("travel_margin_mm", 0.0))
    total_length = tank_length - 2.0 * travel_margin
    if total_length <= 0:
        raise ValueError("Tank travel length must be positive after margins.")

    accel_length = min(float(tank.get("accel_distance_mm", 125.0)), total_length / 2.0)
    cruise_length = max(total_length - 2.0 * accel_length, 0.0)
    cruise_steps = max(1, int(tank.get("cruise_steps", 101)))

    acceleration = (
        target_speed * target_speed / (2.0 * accel_length)
        if accel_length > 0
        else 0.0
    )
    acceleration_time = target_speed / acceleration if acceleration > 0 else 0.0
    cruise_time = cruise_length / target_speed if cruise_length > 0 else 0.0

    if cruise_length > 0 and cruise_time > 0:
        cruise_dt = cruise_time / cruise_steps
        acceleration_steps = max(1, int(round(acceleration_time / cruise_dt)))
        cruise_count = cruise_steps
    else:
        acceleration_steps = max(1, cruise_steps // 2)
        cruise_dt = 0.0
        cruise_count = 0
    acceleration_dt = (
        acceleration_time / acceleration_steps if acceleration_time > 0 else 0.01
    )

    positions = [float(tank.get("x_start_mm", 0.0))]
    durations: list[float] = []
    x_origin = positions[0]

    for index in range(1, acceleration_steps + 1):
        fraction = index / acceleration_steps
        distance = accel_length * fraction * fraction
        positions.append(x_origin + distance)
        durations.append(acceleration_dt)

    for index in range(1, cruise_count + 1):
        local_time = index * cruise_dt
        distance = accel_length + target_speed * local_time
        positions.append(x_origin + distance)
        durations.append(cruise_dt)

    for index in range(1, acceleration_steps + 1):
        fraction = index / acceleration_steps
        distance = accel_length + cruise_length + accel_length * (
            2.0 * fraction - fraction * fraction
        )
        positions.append(x_origin + distance)
        durations.append(acceleration_dt)

    positions[-1] = x_origin + total_length
    return positions, durations


def _axis_word(axis: str, value: float) -> str:
    axis = axis.strip().upper()
    if len(axis) != 1 or not axis.isalpha():
        raise ValueError(f"Invalid CNC axis name '{axis}'.")
    return f"{axis}{value:.3f}"


def _validate_axis(
    axis: str,
    value: float,
    safety: Mapping[str, object],
) -> None:
    minimum_key = f"{axis.lower()}_min_mm"
    maximum_key = f"{axis.lower()}_max_mm"
    if minimum_key not in safety or maximum_key not in safety:
        raise ValueError(
            f"Safety limits '{minimum_key}' and '{maximum_key}' are required "
            f"for configured axis {axis}."
        )
    if value < float(safety[minimum_key]):
        raise ValueError(f"{axis}={value:.3f} mm is below {minimum_key}.")
    if value > float(safety[maximum_key]):
        raise ValueError(f"{axis}={value:.3f} mm is above {maximum_key}.")


def build_tow_program(
    re_value: float,
    h_value: float,
    tank: Mapping[str, object],
    fluid: Mapping[str, object],
    safety: Mapping[str, object],
    z_offset_mm: float = 0.0,
    label: str = "measurement",
) -> TowProgram:
    positions, durations = build_trapezoid_profile(re_value, tank, fluid)
    y_position = tank_y_mm(h_value, tank)
    z_position = float(tank.get("z_mm", 0.0)) + float(z_offset_mm)
    fourth_position = float(tank.get("fourth_mm", -155.0))

    axis_x = str(tank.get("axis_x", "X")).upper()
    axis_y = str(tank.get("axis_y", "Y")).upper()
    axis_z = str(tank.get("axis_z", "Z")).upper()
    axis_fourth = str(tank.get("axis_fourth", "C")).upper()

    for x_position in positions:
        _validate_axis(axis_x, x_position, safety)
    _validate_axis(axis_y, y_position, safety)
    _validate_axis(axis_z, z_position, safety)
    _validate_axis(axis_fourth, fourth_position, safety)

    max_speed = float(safety.get("max_speed_mm_s", float("inf")))
    requested_speed = speed_mm_s(
        re_value,
        length_scale_m=float(fluid.get("length_scale_m", 0.05)),
        viscosity_m2_s=float(fluid.get("kinematic_viscosity_m2_s", 1e-6)),
    )
    if requested_speed > max_speed:
        raise ValueError(
            f"Requested speed {requested_speed:.3f} mm/s exceeds "
            f"max_speed_mm_s={max_speed:.3f}."
        )

    acceleration_distance = min(
        float(tank.get("accel_distance_mm", 125.0)),
        (float(tank["length_mm"]) - 2.0 * float(tank.get("travel_margin_mm", 0.0)))
        / 2.0,
    )
    requested_acceleration = (
        requested_speed**2 / (2.0 * acceleration_distance)
        if acceleration_distance > 0
        else float("inf")
    )
    max_acceleration = float(safety.get("max_acceleration_mm_s2", float("inf")))
    if requested_acceleration > max_acceleration:
        raise ValueError(
            f"Requested acceleration {requested_acceleration:.3f} mm/s^2 exceeds "
            f"max_acceleration_mm_s2={max_acceleration:.3f}."
        )

    coordinate_words = " ".join(
        [
            _axis_word(axis_x, positions[0]),
            _axis_word(axis_y, y_position),
            _axis_word(axis_z, z_position),
            _axis_word(axis_fourth, fourth_position),
        ]
    )
    preposition = (
        f"; ASMR 2.0 {label} pre-position",
        "G21",
        "G90",
        f"G0 {coordinate_words}",
    )

    tow_lines = [
        f"; ASMR 2.0 {label} tow",
        f"; Re={re_value:.8g}, h={h_value:.8g}",
        f"; Z={z_position:.8g} mm",
        "G21",
        "G90",
        "G94",
    ]
    points: list[MotionPoint] = []
    for index in range(1, len(positions)):
        distance = positions[index] - positions[index - 1]
        dt_s = durations[index - 1]
        feed = distance / dt_s * 60.0 if dt_s > 0 else 0.0
        if feed < 0:
            raise ValueError("Generated tow profile contains reverse motion.")
        points.append(MotionPoint(positions[index], dt_s, feed))
        tow_lines.append(f"G1 {_axis_word(axis_x, positions[index])} F{feed:.3f}")
    tow_lines.append("M2")

    return TowProgram(
        label=str(label),
        re_value=float(re_value),
        h_value=float(h_value),
        x_start_mm=float(positions[0]),
        x_end_mm=float(positions[-1]),
        y_mm=float(y_position),
        z_mm=z_position,
        fourth_mm=fourth_position,
        preposition_lines=tuple(preposition),
        tow_lines=tuple(tow_lines),
        points=tuple(points),
    )


def build_paired_tow_programs(
    re_value: float,
    h_value: float,
    tank: Mapping[str, object],
    fluid: Mapping[str, object],
    safety: Mapping[str, object],
) -> tuple[TowProgram, TowProgram]:
    reference_offset = float(tank.get("reference_z_offset_mm", 0.0))
    if abs(reference_offset) <= 1e-9:
        raise ValueError(
            "tank.reference_z_offset_mm must be non-zero for paired pressure "
            "subtraction experiments."
        )
    measurement = build_tow_program(
        re_value,
        h_value,
        tank,
        fluid,
        safety,
        z_offset_mm=0.0,
        label="measurement",
    )
    reference = build_tow_program(
        re_value,
        h_value,
        tank,
        fluid,
        safety,
        z_offset_mm=reference_offset,
        label="reference",
    )
    return measurement, reference


def write_program_files(program: TowProgram, run_directory: Path) -> tuple[Path, Path]:
    run_directory.mkdir(parents=True, exist_ok=True)
    preposition_path = run_directory / "preposition.gcode"
    tow_path = run_directory / "tow.gcode"
    preposition_path.write_text(
        "\n".join(program.preposition_lines) + "\n", encoding="ascii"
    )
    tow_path.write_text("\n".join(program.tow_lines) + "\n", encoding="ascii")
    return preposition_path, tow_path


def executable_gcode(lines: Iterable[str]) -> list[str]:
    commands = []
    for raw in lines:
        command = raw.split(";", 1)[0].strip()
        if command:
            commands.append(command)
    return commands


class SerialCNCController:
    """Small GRBL-style sender that waits for acknowledgement and final Idle."""

    def __init__(self, config: Mapping[str, object], log: LogCallback = print):
        self.config = config
        self.log = log
        self.serial = None
        self._homed = False

    def connect(self) -> None:
        if self.serial is not None:
            return
        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pyserial is required for direct CNC control.") from exc

        port = str(self.config.get("port", "")).strip()
        if not port:
            raise ValueError("cnc.port must be configured.")
        baud = int(self.config.get("baud", 115200))
        self.log(f"[CNC] Opening {port} at {baud} baud")
        self.serial = serial.Serial(
            port=port,
            baudrate=baud,
            timeout=float(self.config.get("read_timeout_s", 0.2)),
            write_timeout=float(self.config.get("write_timeout_s", 2.0)),
        )
        self.serial.write(b"\r\n\r\n")
        self.serial.flush()
        startup_wait = float(self.config.get("startup_wait_s", 2.0))
        deadline = time.monotonic() + startup_wait
        while time.monotonic() < deadline:
            line = self._readline()
            if line:
                self.log(f"[CNC startup] {line}")

    def close(self) -> None:
        if self.serial is not None:
            self.serial.close()
            self.serial = None

    def _readline(self) -> str:
        raw = self.serial.readline()
        return raw.decode("utf-8", errors="replace").strip() if raw else ""

    def send_command(self, command: str, timeout_s: float | None = None) -> None:
        if self.serial is None:
            raise RuntimeError("CNC controller is not connected.")
        command = command.strip()
        if not command:
            return
        timeout_s = (
            float(timeout_s)
            if timeout_s is not None
            else float(self.config.get("ack_timeout_s", 30.0))
        )
        self.serial.write((command + "\n").encode("ascii"))
        self.serial.flush()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            line = self._readline()
            if not line:
                continue
            lower = line.lower()
            if lower.startswith("ok"):
                return
            if lower.startswith("error") or lower.startswith("alarm"):
                raise RuntimeError(f"Controller rejected '{command}': {line}")
            self.log(f"[CNC] {line}")
        raise TimeoutError(f"Timed out waiting for acknowledgement of '{command}'.")

    def status(self) -> str | None:
        if self.serial is None:
            raise RuntimeError("CNC controller is not connected.")
        self.serial.write(b"?")
        self.serial.flush()
        deadline = time.monotonic() + float(self.config.get("status_timeout_s", 2.0))
        while time.monotonic() < deadline:
            line = self._readline()
            if line.startswith("<") and line.endswith(">"):
                return line[1:-1].split("|", 1)[0]
            if line.lower().startswith(("alarm", "error")):
                raise RuntimeError(f"Controller status error: {line}")
        return None

    def wait_idle(self, timeout_s: float | None = None) -> None:
        timeout_s = (
            float(timeout_s)
            if timeout_s is not None
            else float(self.config.get("motion_timeout_s", 300.0))
        )
        poll_interval = float(self.config.get("status_poll_interval_s", 0.2))
        deadline = time.monotonic() + timeout_s
        last_state = None
        while time.monotonic() < deadline:
            state = self.status()
            if state and state != last_state:
                self.log(f"[CNC] State: {state}")
                last_state = state
            if state == "Idle":
                return
            if state == "Alarm":
                raise RuntimeError("Controller entered Alarm state.")
            time.sleep(poll_interval)
        raise TimeoutError("Timed out waiting for controller Idle state.")

    def home_once(self) -> None:
        if self._homed:
            return
        if bool(self.config.get("unlock_before_home", True)):
            self.send_command("$X")
        self.log("[CNC] Homing all axes")
        self.send_command("$H", timeout_s=float(self.config.get("home_timeout_s", 180.0)))
        self.wait_idle(timeout_s=float(self.config.get("home_timeout_s", 180.0)))
        self._homed = True

    def send_program(
        self,
        lines: Iterable[str],
        wait_for_idle: bool = True,
        motion_timeout_s: float | None = None,
    ) -> None:
        commands = executable_gcode(lines)
        for index, command in enumerate(commands, start=1):
            self.send_command(command)
            if index == len(commands) or index % 10 == 0:
                self.log(f"[CNC] Sent {index}/{len(commands)} commands")
        if wait_for_idle:
            self.wait_idle(timeout_s=motion_timeout_s)

    def preposition(self, program: TowProgram) -> None:
        self.log(
            f"[CNC] Pre-positioning {program.label} tow at "
            f"X={program.x_start_mm:.3f}, Y={program.y_mm:.3f}, "
            f"Z={program.z_mm:.3f}"
        )
        self.send_program(program.preposition_lines, wait_for_idle=True)

    def tow(self, program: TowProgram) -> None:
        self.log(
            f"[CNC] Starting {program.label} tow "
            f"X={program.x_start_mm:.3f} -> {program.x_end_mm:.3f}, "
            f"Z={program.z_mm:.3f}"
        )
        self.send_program(program.tow_lines, wait_for_idle=True)

    def feed_hold(self) -> None:
        if self.serial is None:
            return
        try:
            self.serial.write(b"!")
            self.serial.flush()
            self.log("[CNC] Feed hold sent")
        except Exception as exc:  # noqa: BLE001 - best-effort emergency action.
            self.log(f"[CNC] Could not send feed hold: {exc}")

    def __enter__(self) -> "SerialCNCController":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
