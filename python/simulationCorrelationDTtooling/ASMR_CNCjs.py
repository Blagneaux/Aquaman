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
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ----------------------------
# Hardcoded test configuration
# ----------------------------
TANK_LENGTH_MM = 750.0
TANK_WIDTH_MM = 690.0
TRAVEL_MM = 300.0  # 50 cm
MAX_SPEED_M_S = 0.01

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
        "$X",
        "$H",  # Home all axes
        "G21",  # mm units
        "G90",  # absolute coordinates
        "G94",  # feed per minute
        f"G0 X{X_START_MM:.3f} Y{Y_CENTER_MM:.3f} Z{Z_MM:.3f}",
        f"G1 X{X_END_MM:.3f} Y{Y_CENTER_MM:.3f} Z{Z_MM:.3f} F{feed_mm_min:.3f}",
        "M2",
    ]


def write_gcode_file(output_path: Path) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(build_gcode()) + "\n", encoding="ascii")
    return output_path


def detect_cli_mode(cli: str) -> str:
    """
    Classify the executable:
    - 'server': cncjs/cnc binaries that start the web server.
    - 'sender': tools that accept serial args (-p/-b) for direct streaming.
    """
    exe = Path(cli).name.lower()
    if exe.endswith(".exe"):
        exe = exe[:-4]
    if exe in {"cncjs", "cnc"}:
        return "server"
    return "sender"


def build_sender_command(
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


def build_server_command(
    gcode_path: Path,
    controller: str | None,
    cli_cmd: list[str],
    extra_args: list[str] | None = None,
) -> list[str]:
    # Expose the G-code output directory in CNCjs watch-directory file picker.
    cmd = list(cli_cmd) + ["--watch-directory", str(gcode_path.parent)]
    if controller:
        cmd += ["--controller", controller]
    if extra_args:
        cmd += extra_args
    return cmd


def parse_gcode_commands(gcode_path: Path) -> list[str]:
    """
    Load a G-code file and return executable commands:
    - strips comments after ';'
    - strips blank lines
    """
    commands: list[str] = []
    for raw_line in gcode_path.read_text(encoding="ascii", errors="ignore").splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if line:
            commands.append(line)
    return commands


def run_sender_pyserial(
    gcode_path: Path,
    port: str,
    baud: int,
    startup_wait_s: float,
    ack_timeout_s: float,
) -> None:
    try:
        import serial  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: pyserial. Install it with `pip install pyserial` in your aquaman environment."
        ) from exc

    commands = parse_gcode_commands(gcode_path)
    if not commands:
        raise RuntimeError(f"No executable G-code commands found in '{gcode_path}'.")

    print(f"[INFO] Sender backend: pyserial ({len(commands)} commands)")
    print(f"[INFO] Opening serial port: {port} @ {baud}")

    with serial.Serial(port=port, baudrate=baud, timeout=0.2, write_timeout=2.0) as ser:
        # Wake up common GRBL-like controllers and capture startup lines.
        ser.write(b"\r\n\r\n")
        ser.flush()
        startup_deadline = time.monotonic() + startup_wait_s
        while time.monotonic() < startup_deadline:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                print(f"[SERIAL] {line}")

        for idx, cmd in enumerate(commands, start=1):
            payload = (cmd + "\n").encode("ascii")
            ser.write(payload)
            ser.flush()

            deadline = time.monotonic() + ack_timeout_s
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Timeout waiting for controller reply after command #{idx}: {cmd}"
                    )
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                low = line.lower()
                if low.startswith("ok"):
                    break
                if low.startswith("error") or low.startswith("alarm"):
                    raise RuntimeError(f"Controller rejected command #{idx} '{cmd}': {line}")
                print(f"[SERIAL] {line}")

            if idx == len(commands) or idx % 5 == 0:
                print(f"[INFO] Sent {idx}/{len(commands)} commands")


def resolve_cncjs_cli(cli: str) -> str:
    """
    Resolve the CNCjs CLI executable.
    - If --cli is set to "auto", try common executable names.
    - Otherwise, use the provided name/path as-is after existence check.
    """
    if cli == "auto":
        for candidate in ("cnc", "cncjs", "cncjs-cli"):
            if shutil.which(candidate):
                print(f"[INFO] Using CNCjs CLI: {candidate}")
                return candidate
        raise FileNotFoundError(
            "Could not find CNCjs CLI executable. Tried: cnc, cncjs, cncjs-cli. "
            "Install CNCjs globally or pass --cli with the full executable path."
        )

    # Support either an executable on PATH or an explicit filesystem path.
    if shutil.which(cli) or Path(cli).exists():
        return cli

    raise FileNotFoundError(
        f"Could not find CLI executable '{cli}'. "
        "Pass --cli with a valid executable name/path."
    )


def resolve_headless_server_launcher(cli: str) -> list[str]:
    """
    Build a command prefix for headless CNCjs server mode (no Electron UI).
    Attempts to run bundled server-cli.js via `node`.
    """
    node = shutil.which("node")
    if not node:
        raise FileNotFoundError(
            "Headless mode needs Node.js (`node`) on PATH to run server-cli.js. "
            "Install Node.js, or run without --no-ui."
        )

    cli_path_str = shutil.which(cli)
    cli_path = Path(cli_path_str).resolve() if cli_path_str else None
    candidates: list[Path] = []

    if cli_path:
        # Typical desktop-app layout: <install_root>/resources/app/server-cli.js
        candidates.append(cli_path.parent / "resources" / "app" / "server-cli.js")

    if sys.platform.startswith("win"):
        candidates.append(
            Path.home() / "AppData" / "Local" / "Programs" / "cncjs-app" / "resources" / "app" / "server-cli.js"
        )

    for candidate in candidates:
        if candidate.exists():
            server_cli = str(candidate)
            # server-cli.js exports a function; it is not directly executable as a script.
            # Run it via `node -e` and invoke the exported function.
            bootstrap = (
                "const run=require(process.argv[1]);"
                "run().catch((err)=>{console.error(err);process.exit(1);});"
            )
            return [node, "-e", bootstrap, server_cli]

    raise FileNotFoundError(
        "Could not locate CNCjs server-cli.js for headless mode. "
        "Pass --cli with a headless server executable, or run without --no-ui."
    )


def parse_args() -> argparse.Namespace:
    default_output = Path("python/simulationCorrelationDTtooling/cncjs_tests/tank_centerline_50cm_0p02ms.gcode")
    parser = argparse.ArgumentParser(
        description="Standalone CNCjs tank-motion test for ASMR."
    )
    parser.add_argument("--port", type=str, default=None, help="Serial port (e.g., COM5, /dev/ttyUSB0).")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate.")
    parser.add_argument("--controller", type=str, default=None, help="Optional CNCjs controller/mode.")
    parser.add_argument(
        "--cli",
        type=str,
        default="auto",
        help='CNCjs executable name/path, or "auto" to try cnc, cncjs, then cncjs-cli.',
    )
    parser.add_argument(
        "--cli-mode",
        type=str,
        default="auto",
        choices=("auto", "sender", "server"),
        help='How to interpret CLI options: "sender" uses -p/-b for serial streaming, '
             '"server" starts CNCjs web server mode, "auto" detects from executable name.',
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Run without UI. With --port, this prefers automatic sender mode.",
    )
    parser.add_argument(
        "--sender-backend",
        type=str,
        default="auto",
        choices=("auto", "cli", "pyserial"),
        help='Sender mode backend: "cli" uses sender executable, "pyserial" streams directly, "auto" picks best option.',
    )
    parser.add_argument(
        "--startup-wait",
        type=float,
        default=2.0,
        help="Seconds to wait for controller startup messages in pyserial sender mode.",
    )
    parser.add_argument(
        "--ack-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for an OK/ERROR reply per command in pyserial sender mode.",
    )
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
    if args.startup_wait < 0:
        raise ValueError("--startup-wait must be >= 0")
    if args.ack_timeout <= 0:
        raise ValueError("--ack-timeout must be > 0")

    gcode_path = write_gcode_file(args.out)
    print(f"[INFO] G-code written to: {gcode_path}")
    print(f"[INFO] Test path: X {X_START_MM:.1f} -> {X_END_MM:.1f} mm, Y {Y_CENTER_MM:.1f} mm")
    print(f"[INFO] Max speed: {MAX_SPEED_M_S:.4f} m/s")

    # Convenience: fully automatic headless motion with no-ui + serial port.
    force_pyserial_sender = (
        args.no_ui and args.port is not None and args.cli_mode == "auto" and args.sender_backend in {"auto", "pyserial"}
    )

    cli: str | None = None
    detected_cli_mode: str | None = None
    if force_pyserial_sender:
        cli_mode = "sender"
        sender_backend = "pyserial"
        print("[INFO] --no-ui with --port detected: forcing sender mode with pyserial backend.")
    else:
        skip_cli_resolution = args.cli_mode == "sender" and args.sender_backend == "pyserial"
        if not skip_cli_resolution:
            try:
                cli = resolve_cncjs_cli(args.cli)
            except FileNotFoundError as exc:
                print(f"[ERROR] {exc}")
                return 2
            detected_cli_mode = detect_cli_mode(cli)

        cli_mode = args.cli_mode if args.cli_mode != "auto" else (detected_cli_mode or "sender")
        if args.sender_backend == "auto":
            sender_backend = "cli" if detected_cli_mode == "sender" else "pyserial"
        else:
            sender_backend = args.sender_backend

    print(f"[INFO] CLI mode: {cli_mode}")

    if cli_mode == "sender":
        if not args.port:
            raise ValueError("Missing --port in sender mode. Use --dry-run if you only want to generate G-code.")
        print(f"[INFO] Sender backend: {sender_backend}")

        if sender_backend == "cli":
            if cli is None:
                try:
                    cli = resolve_cncjs_cli(args.cli)
                except FileNotFoundError as exc:
                    print(f"[ERROR] {exc}")
                    return 2
            if detect_cli_mode(cli) == "server":
                print(
                    f"[ERROR] '{cli}' is a server-style CNCjs executable, not a sender CLI. "
                    "Use --sender-backend pyserial for fully automatic no-UI motion."
                )
                return 2

            cmd = build_sender_command(
                gcode_path=gcode_path,
                port=args.port,
                baud=args.baud,
                controller=args.controller,
                cli=cli,
                extra_args=args.extra_arg,
            )
            if args.dry_run:
                print(f"[DRY RUN] Command: {' '.join(cmd)}")
                return 0

            print(f"[INFO] Running: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except FileNotFoundError as exc:
                print(f"[ERROR] Failed to start '{cmd[0]}': {exc}")
                return 2
            print("[INFO] Sender command completed.")
            return 0

        if args.dry_run:
            print(
                f"[DRY RUN] pyserial sender would stream '{gcode_path}' to port '{args.port}' at {args.baud} baud."
            )
            return 0

        try:
            run_sender_pyserial(
                gcode_path=gcode_path,
                port=args.port,
                baud=args.baud,
                startup_wait_s=args.startup_wait,
                ack_timeout_s=args.ack_timeout,
            )
        except (RuntimeError, TimeoutError) as exc:
            print(f"[ERROR] {exc}")
            return 2
        print("[INFO] pyserial sender completed.")
        return 0

    if cli is None:
        try:
            cli = resolve_cncjs_cli(args.cli)
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}")
            return 2

    cli_cmd = [cli]
    if args.no_ui:
        exe = Path(cli).name.lower()
        if exe.endswith(".exe"):
            exe = exe[:-4]
        if exe == "cnc":
            print("[INFO] '--cli cnc' is already headless; no UI will be opened by this script.")
        else:
            try:
                cli_cmd = resolve_headless_server_launcher(cli)
                print(f"[INFO] Headless launcher: {' '.join(cli_cmd)}")
            except FileNotFoundError as exc:
                print(f"[ERROR] {exc}")
                return 2

    cmd = build_server_command(
        gcode_path=gcode_path,
        controller=args.controller,
        cli_cmd=cli_cmd,
        extra_args=args.extra_arg,
    )
    if args.port:
        print(
            f"[WARN] --port {args.port} is a serial port, but '{cli}' in server mode uses -p/--port as web listen port."
        )
        if args.no_ui:
            print(
                f"[INFO] Headless server mode started. This script does not auto-connect '{args.port}' "
                f"or auto-run '{gcode_path.name}'. Use sender mode for direct streaming."
            )
        else:
            print(
                f"[INFO] In CNCjs UI, connect manually to serial port '{args.port}' at baud {args.baud} and run '{gcode_path.name}'."
            )
    else:
        print(
            f"[INFO] In CNCjs UI, connect to your serial device and run '{gcode_path.name}' "
            f"from watch directory '{gcode_path.parent}'."
        )

    if args.dry_run:
        print(f"[DRY RUN] Command: {' '.join(cmd)}")
        return 0

    print(f"[INFO] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        print(f"[ERROR] Failed to start '{cmd[0]}': {exc}")
        return 2
    print("[INFO] CNCjs command completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
