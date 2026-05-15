#!/usr/bin/env python3
"""
CLI: detect and label RS-232 sensors.

Usage:
  python detect_sensors.py              # scan once and show results
  python detect_sensors.py --monitor    # watch for plug/unplug events
  python detect_sensors.py --list       # show saved sensor signatures
  python detect_sensors.py --label      # interactively label an unknown port
"""

import argparse
import re
import sys
from pathlib import Path

# pyserial check must come before any src import (src.serial_detector imports serial)
try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("pyserial not installed. Run: pip install pyserial")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from src.serial_detector import (
    SIGNATURES_FILE,
    load_signatures,
    monitor,
    probe_port,
    save_signature,
    scan_ports,
)


def cmd_scan():
    print("=" * 60)
    print("RS-232 Sensor Detection — one-shot scan")
    print("=" * 60)
    results = scan_ports(verbose=True)
    if not results:
        return

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    unknown = []
    for r in results:
        sensor = r.get("sensor_type") or "UNKNOWN"
        print(f"  {r['device']:20s}  {sensor}")
        if not r.get("sensor_type") and r.get("raw"):
            unknown.append(r)

    if unknown:
        print(f"\n{len(unknown)} unrecognised sensor(s). Run with --label to name them.")


def cmd_monitor():
    def on_event(event, info):
        if event == "added" and not info.get("sensor_type") and info.get("raw"):
            ans = input(
                f"\n  Unknown sensor on {info['device']}. "
                "Enter sensor type (pH/DO/Temperature/Foam/other) or blank to skip: "
            ).strip()
            if ans:
                _save_from_probe(info, ans)

    monitor(on_change=on_event)


def cmd_label():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return

    print("Available ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device}  ({p.description or 'no description'})")

    idx = input("Select port number: ").strip()
    if not idx.isdigit() or int(idx) >= len(ports):
        print("Invalid selection.")
        return

    port = ports[int(idx)]
    print(f"Probing {port.device} ...")
    probe = probe_port(port.device, vid=port.vid, pid=port.pid)

    if not probe:
        print("No response from device.")
        return

    print(f"  Baud: {probe['baud']}  Config: {probe['config']}")
    print(f"  Raw response: {repr(probe['raw'][:200])}")

    sensor_type = input(
        "Enter sensor type label (e.g. pH, DO, Temperature, Foam): "
    ).strip()
    if not sensor_type:
        print("Cancelled.")
        return

    # Auto-derive a sensible default pattern from the raw response
    default_pattern = re.escape(probe["raw"][:8].strip()) if probe["raw"] else ".*"
    pattern = input(
        f"Enter regex pattern (or blank for auto: {repr(default_pattern)}): "
    ).strip() or default_pattern

    entry = {
        "device": port.device,
        "description": port.description or "",
        "vid": port.vid,
        "pid": port.pid,
        "baud": probe["baud"],
        "config": probe.get("config"),
        "raw_sample": probe["raw"][:200],
        "sensor_type": sensor_type,
        "pattern": pattern,
    }
    save_signature(entry)
    print(f"Saved signature for '{sensor_type}' → {SIGNATURES_FILE}")
    print(f"  Pattern: {pattern}")


def _save_from_probe(info: dict, sensor_type: str):
    raw = info.get("raw") or ""
    default_pattern = re.escape(raw[:8].strip()) if raw else ".*"
    pattern = input(
        f"  Enter regex pattern (or blank for auto: {repr(default_pattern)}): "
    ).strip() or default_pattern
    entry = {**info, "sensor_type": sensor_type, "pattern": pattern}
    save_signature(entry)
    print(f"  Saved as '{sensor_type}' (pattern: {pattern}).")


def cmd_list():
    sigs = load_signatures()
    if not sigs:
        print("No saved signatures yet. Run --label to add one.")
        return
    print(f"Saved signatures ({SIGNATURES_FILE}):\n")
    for s in sigs:
        print(f"  Sensor : {s.get('sensor_type')}")
        print(f"  Pattern: {s.get('pattern')}")
        print(f"  Port   : {s.get('device')}  Baud: {s.get('baud')}  Config: {s.get('config', '—')}")
        print(f"  Sample : {repr(s.get('raw_sample', '')[:80])}")
        print()


def main():
    parser = argparse.ArgumentParser(description="RS-232 sensor detector")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--monitor", action="store_true", help="Watch for plug/unplug events")
    group.add_argument("--label", action="store_true", help="Interactively label an unknown port")
    group.add_argument("--list", action="store_true", help="List saved sensor signatures")
    args = parser.parse_args()

    if args.monitor:
        cmd_monitor()
    elif args.label:
        cmd_label()
    elif args.list:
        cmd_list()
    else:
        cmd_scan()


if __name__ == "__main__":
    main()
#review 