#!/usr/bin/env python3
"""
ASMR_CNCjs.py

Standalone CNCjs tank-motion test:
- Straight line trajectory.
- Center of the tank (fixed Y).
- 50 cm travel.
- Max speed = 0.02 m/s.

Usage examples:
  python python/simulationCorrelationDTtooling/ASMR_CNCjs.py --dry-run
  python python/simulationCorrelationDTtooling/ASMR_CNCjs.py --port COM5 --baud 115200
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


# ----------------------------
# Hardcoded test configuration
# ----------------------------
TANK_LENGTH_MM = 750.0
TANK_WIDTH_MM = 690.0
TRAVEL_MM = 500.0  # 50 cm
MAX_SPEED_M_S = 0.02

# Centered 50 cm line in a 75 cm tank -> start at 12.5 cm, end at 62.5 cm.
X_START_MM = (TANK_LENGTH_MM - TRAVEL_MM) / 2.0
X_END_MM = X_START_MM + TRAVEL_MM

# Centerline of the tank in machine coordinates for this test.
# If your machine origin is not centered in Y, set this constant accordingly.
Y_CENTER_MM = 0.0

# Optional fixed Z (and C if needed later).
Z_MM = 0.0


def build_gcode() -> list[str]:
    feed_mm_min = MAX_SPEED_M_S * 1000.0 * 60.0
    return [
        "; ASMR CNCjs standalone tank test",
        f"; Tank: {TANK_LENGTH_MM:.1f} x {TANK_WIDTH_MM:.1f} mm",
        f"; Path: X {X_START_MM:.1f} -> {X_END_MM:.1f} mm at Y {Y_CENTER_MM:.1f} mm",
        f"; Max speed: {MAX_SPEED_M_S:.4f} m/s ({feed_mm_min:.1f} mm/min)",
        "G21",  # mm units
        "G90",  # absolute coordinates
        "G94",  # feed per minute
        f"G0 X{X_START_MM:.3f} Y{Y_CENTER_MM:.3f} Z{Z_MM:.3f}",
        f"G1 X{X_END_MM:.3f} Y{Y_CENTER_MM:.3f} Z{Z_MM:.3f} F{feed_mm_min:.3f}",
        "M2",
    ]


def write_gcode_file(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(build_gcode()) + "\n", encoding="ascii")
    return output_path


def build_cncjs_command(
    gcode_path: Path,
    port: str,
    baud: int,
    controller: str | None,
    cli: str,
    extra_args: list[str] | None = None,
) -> list[str]:
    cmd = [cli, "-p", port, "-b", str(baud)]
    if controller:
        cmd += ["-m", controller]
    if extra_args:
        cmd += extra_args
    cmd.append(str(gcode_path))
    return cmd


def parse_args() -> argparse.Namespace:
    default_output = Path("python/simulationCorrelationDTtooling/cncjs_tests/tank_centerline_50cm_0p02ms.gcode")
    parser = argparse.ArgumentParser(
        description="Standalone CNCjs tank-motion test for ASMR."
    )
    parser.add_argument("--port", type=str, default=None, help="Serial port (e.g., COM5, /dev/ttyUSB0).")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate.")
    parser.add_argument("--controller", type=str, default=None, help="Optional CNCjs controller/mode.")
    parser.add_argument("--cli", type=str, default="cncjs", help="CNCjs CLI executable name/path.")
    parser.add_argument("--out", type=Path, default=default_output, help="Output G-code file path.")
    parser.add_argument("--dry-run", action="store_true", help="Only generate G-code and print command.")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional CNCjs CLI argument (repeatable).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    gcode_path = write_gcode_file(args.out)
    print(f"[INFO] G-code written to: {gcode_path}")
    print(f"[INFO] Test path: X {X_START_MM:.1f} -> {X_END_MM:.1f} mm, Y {Y_CENTER_MM:.1f} mm")
    print(f"[INFO] Max speed: {MAX_SPEED_M_S:.4f} m/s")

    if args.dry_run:
        if args.port:
            cmd = build_cncjs_command(
                gcode_path=gcode_path,
                port=args.port,
                baud=args.baud,
                controller=args.controller,
                cli=args.cli,
                extra_args=args.extra_arg,
            )
            print(f"[DRY RUN] Command: {' '.join(cmd)}")
        else:
            print("[DRY RUN] No --port provided, so no CNCjs command was built.")
        return 0

    if not args.port:
        raise ValueError("Missing --port. Use --dry-run if you only want to generate G-code.")

    cmd = build_cncjs_command(
        gcode_path=gcode_path,
        port=args.port,
        baud=args.baud,
        controller=args.controller,
        cli=args.cli,
        extra_args=args.extra_arg,
    )
    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("[INFO] CNCjs command completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
