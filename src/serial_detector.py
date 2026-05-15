"""
RS-232 sensor auto-detection — with legacy sensor support.

Detection flow per port:
  1. Fast path: try baud+config from saved signature for this vid/pid first
  2. Full scan: try every (baud, config) combination
  3. For each combo: open port → warm-up delay → passive listen → active probe
  4. Skip garbled data (wrong baud/parity) via readability check
  5. Match readable response against saved signatures and built-in patterns
"""

import json
import re
import time
from pathlib import Path
from typing import Optional

import serial
import serial.tools.list_ports

SIGNATURES_FILE = Path(__file__).parent.parent / "data" / "sensor_signatures.json"

# Slow legacy rates first so old sensors are found before giving up
BAUD_RATES = [9600, 4800, 19200, 2400, 38400, 1200, 57600, 115200, 600, 300]

# (label, bytesize, parity, stopbits) — 8N1 most common, 7E1 common in old industrial gear
SERIAL_CONFIGS = [
    ("8N1", serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE),
    ("7E1", serial.SEVENBITS, serial.PARITY_EVEN, serial.STOPBITS_ONE),
    ("7O1", serial.SEVENBITS, serial.PARITY_ODD,  serial.STOPBITS_ONE),
    ("8E1", serial.EIGHTBITS, serial.PARITY_EVEN, serial.STOPBITS_ONE),
]

WARMUP_DELAY = 0.5  # seconds after port open — lets old hardware wake up

PROBE_COMMANDS = [b"\r\n", b"?\r\n", b"R\r\n", b"M\r\n", b"D\r\n", b"S\r\n", b"#01\r\n"]

BUILTIN_PATTERNS: list[dict] = [
    {
        "sensor_type": "pH",
        "description": "Generic pH sensor",
        "pattern": r"pH\s*[=:]\s*[\d.]+",
        "flags": re.IGNORECASE,
    },
    {
        "sensor_type": "DO",
        "description": "Dissolved oxygen sensor",
        "pattern": r"DO\s*[=:]\s*[\d.]+|O2\s*[=:]\s*[\d.]+|air\s*sat",
        "flags": re.IGNORECASE,
    },
    {
        "sensor_type": "Temperature",
        "description": "Temperature probe",
        # Require explicit T= prefix OR decimal number + °C to avoid false matches like "5C"
        "pattern": r"T(emp)?\s*[=:]\s*[\d.]+|\d+\.\d+\s*°?C\b",
        "flags": re.IGNORECASE,
    },
    {
        "sensor_type": "Foam",
        "description": "Foam / level sensor",
        "pattern": r"foam|level|F[=:]\s*[01]",
        "flags": re.IGNORECASE,
    },
    {
        "sensor_type": "Stirrer",
        "description": "Agitation / RPM sensor",
        "pattern": r"rpm\s*[=:]\s*[\d.]+|stir",
        "flags": re.IGNORECASE,
    },
    {
        "sensor_type": "Flow",
        "description": "Flow meter",
        "pattern": r"flow\s*[=:]\s*[\d.]+|ml/min|L/h",
        "flags": re.IGNORECASE,
    },
]


# ─── Readability check ────────────────────────────────────────────────────────

def is_readable(raw_bytes: bytes) -> bool:
    """True if byte stream looks like valid ASCII/Latin-1 sensor output (not garbled)."""
    if not raw_bytes:
        return False
    printable = sum(1 for b in raw_bytes if 32 <= b <= 126 or b in (9, 10, 13))
    return printable / len(raw_bytes) > 0.7


# ─── Signature persistence (with mtime cache) ────────────────────────────────

_sig_cache: list[dict] = []
_sig_mtime: float = 0.0


def load_signatures() -> list[dict]:
    global _sig_cache, _sig_mtime
    try:
        mtime = SIGNATURES_FILE.stat().st_mtime
    except FileNotFoundError:
        return []
    if mtime != _sig_mtime:
        try:
            _sig_cache = json.loads(SIGNATURES_FILE.read_text())
            _sig_mtime = mtime
        except json.JSONDecodeError:
            bad = SIGNATURES_FILE.with_suffix(".bak")
            SIGNATURES_FILE.rename(bad)
            print(f"WARNING: signatures file corrupted, backed up to {bad}")
            _sig_cache = []
    return _sig_cache


def _dedup_key(entry: dict) -> tuple:
    """Dedup key for save_signature. Uses device path when vid is None (native RS-232)."""
    if entry.get("vid") is None:
        return (None, None, entry.get("device"), entry.get("baud"), entry.get("config"))
    return (entry.get("vid"), entry.get("pid"), entry.get("baud"), entry.get("config"))


def save_signature(entry: dict) -> None:
    global _sig_mtime
    sigs = load_signatures()
    sigs = [s for s in sigs if _dedup_key(s) != _dedup_key(entry)]
    sigs.append(entry)
    SIGNATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SIGNATURES_FILE.write_text(json.dumps(sigs, indent=2))
    _sig_mtime = 0.0  # invalidate cache


# ─── Response matching ────────────────────────────────────────────────────────

def match_response(raw: str, vid=None, pid=None) -> Optional[str]:
    """
    Return sensor type if raw response matches a known pattern.

    Saved signatures with pattern '.*' are scoped to the same hardware (vid+pid)
    so they don't hijack detection for all future sensors.
    """
    for sig in load_signatures():
        pattern = sig.get("pattern") or ""
        if not pattern:
            continue
        if pattern == ".*":
            # Wildcard signatures only match if hardware ID is the same
            if sig.get("vid") == vid and sig.get("pid") == pid:
                return sig["sensor_type"]
        elif re.search(pattern, raw, re.IGNORECASE):
            return sig["sensor_type"]

    for bp in BUILTIN_PATTERNS:
        if re.search(bp["pattern"], raw, bp["flags"]):
            return bp["sensor_type"]

    return None


# ─── Serial probing ───────────────────────────────────────────────────────────

def _try_one_config(
    device: str, baud: int, cfg_name: str,
    bytesize, parity, stopbits,
    passive_timeout: float,
    vid=None, pid=None,
) -> Optional[dict]:
    """
    Open port with one (baud, config) combination.
    Phase 1: passive listen. Phase 2: active probe if nothing arrived.
    Returns result dict or None.
    """
    try:
        with serial.Serial(
            device,
            baudrate=baud,
            bytesize=bytesize,
            parity=parity,
            stopbits=stopbits,
            timeout=passive_timeout,
        ) as ser:
            time.sleep(WARMUP_DELAY)
            ser.reset_input_buffer()

            # Phase 1: passive listen — many old sensors stream without being asked
            raw_bytes = ser.read(256)

            if raw_bytes and not is_readable(raw_bytes):
                return None  # garbled → wrong baud or parity, skip immediately

            if not raw_bytes:
                # Phase 2: active probe
                ser.timeout = 0.5
                for cmd in PROBE_COMMANDS:
                    ser.write(cmd)
                    time.sleep(0.2)
                    waiting = ser.in_waiting
                    if waiting == 0:
                        continue  # nothing came back, try next command
                    chunk = ser.read(waiting)
                    if chunk:
                        raw_bytes += chunk
                        if is_readable(raw_bytes):
                            break  # readable response → stop probing

            if not raw_bytes or not is_readable(raw_bytes):
                return None

            raw = raw_bytes.decode("latin-1", errors="replace").strip()
            return {
                "baud": baud,
                "config": cfg_name,
                "raw": raw,
                "sensor_type": match_response(raw, vid=vid, pid=pid),
            }

    except serial.SerialException:
        return None


def probe_port(device: str, vid=None, pid=None) -> Optional[dict]:
    """
    Probe `device` for a sensor response.

    Fast path: try baud+config from any saved signature matching this vid/pid first.
    Full scan: try all (baud, config) combinations if fast path misses.
    """
    # Fast path — saved signature for this hardware
    if vid is not None or pid is not None:
        for sig in load_signatures():
            if sig.get("vid") == vid and sig.get("pid") == pid:
                saved_baud = sig.get("baud")
                saved_cfg  = sig.get("config", "8N1")
                cfg_entry  = next((c for c in SERIAL_CONFIGS if c[0] == saved_cfg), None)
                if saved_baud and cfg_entry:
                    passive_timeout = 2.5 if saved_baud <= 1200 else 1.5
                    result = _try_one_config(
                        device, saved_baud, cfg_entry[0],
                        cfg_entry[1], cfg_entry[2], cfg_entry[3],
                        passive_timeout, vid=vid, pid=pid,
                    )
                    if result:
                        return result

    # Full scan
    for baud in BAUD_RATES:
        passive_timeout = 2.5 if baud <= 1200 else 1.5
        for cfg_name, bytesize, parity, stopbits in SERIAL_CONFIGS:
            result = _try_one_config(
                device, baud, cfg_name, bytesize, parity, stopbits,
                passive_timeout, vid=vid, pid=pid,
            )
            if result:
                return result

    return None


# ─── Port scanning ────────────────────────────────────────────────────────────

def scan_ports(verbose: bool = True) -> list[dict]:
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        if verbose:
            print("No serial ports found.")
        return []

    results = []
    for port in ports:
        entry = {
            "device": port.device,
            "description": port.description or "",
            "vid": port.vid,
            "pid": port.pid,
            "sensor_type": None,
            "baud": None,
            "config": None,
            "raw": None,
        }

        if verbose:
            print(f"\nProbing {port.device} ({port.description}) ...")

        probe = probe_port(port.device, vid=port.vid, pid=port.pid)
        if probe:
            entry.update(probe)
            if verbose:
                sensor = entry["sensor_type"] or "UNKNOWN"
                print(f"  {probe['baud']} {probe['config']}  →  Sensor: {sensor}")
                print(f"  Raw: {repr(probe['raw'][:120])}")
        else:
            if verbose:
                print("  No response at any baud rate or serial config.")

        results.append(entry)

    return results


# ─── Continuous hot-plug monitor ──────────────────────────────────────────────

def monitor(poll_interval: float = 2.0, on_change=None):
    """
    Watch for serial ports being plugged/unplugged.
    Calls on_change(event, port_info) where event is 'added' or 'removed'.
    Runs forever (Ctrl-C to stop).
    """
    known = {p.device for p in serial.tools.list_ports.comports()}
    print(f"Monitoring serial ports (polling every {poll_interval}s) — Ctrl-C to stop")

    while True:
        time.sleep(poll_interval)
        current_ports = {p.device: p for p in serial.tools.list_ports.comports()}
        current = set(current_ports.keys())

        for dev in known - current:
            print(f"\n[DISCONNECTED] {dev}")
            if on_change:
                on_change("removed", {"device": dev})

        for dev in current - known:
            print(f"\n[CONNECTED] {dev}")
            port_info = current_ports[dev]
            probe = probe_port(dev, vid=port_info.vid, pid=port_info.pid)
            result = {
                "device": dev,
                "description": port_info.description or "",
                "vid": port_info.vid,
                "pid": port_info.pid,
                **(probe or {}),
            }
            sensor = result.get("sensor_type") or "UNKNOWN"
            print(f"  Sensor: {sensor}")
            if probe:
                print(f"  Baud: {probe['baud']} {probe['config']}  Raw: {repr(probe['raw'][:120])}")
            if on_change:
                on_change("added", result)

        known = current
