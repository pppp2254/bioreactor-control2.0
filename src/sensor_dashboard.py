"""
Dash dashboard for RS-232 sensor detection.

Shows:
- Live list of connected serial ports, refreshing every 3 s
- Detected sensor type, baud rate, serial config, and raw response preview
- Connection status badge (connected / probing / no response / disconnected)
- Label input for UNKNOWN sensors — saves signature for future auto-recognition

Run:  python src/sensor_dashboard.py
  OR: python -m src.sensor_dashboard  (from project root)
"""

import copy
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path so `src.serial_detector` resolves correctly
# regardless of whether this file is run as `python src/sensor_dashboard.py`
# or `python -m src.sensor_dashboard`.
sys.path.insert(0, str(Path(__file__).parent.parent))

import serial.tools.list_ports
from dash import ALL, Dash, Input, Output, State, callback_context, dcc, html, no_update

from src.serial_detector import probe_port, save_signature

# ── Shared state ──────────────────────────────────────────────────────────────
_lock = threading.Lock()
_port_state: dict[str, dict] = {}   # device → info dict
_executor = ThreadPoolExecutor(max_workers=4)


def _make_entry(port, status: str) -> dict:
    return {
        "device": port.device,
        "description": port.description or "",
        "vid": port.vid,
        "pid": port.pid,
        "status": status,
        "sensor_type": None,
        "baud": None,
        "config": None,
        "raw": None,
        "updated": datetime.now().strftime("%H:%M:%S"),
    }


def _probe_and_update(device: str):
    """Run in a worker thread: probe device, update shared state."""
    with _lock:
        info = _port_state.get(device, {})
    vid = info.get("vid")
    pid = info.get("pid")

    probe = probe_port(device, vid=vid, pid=pid)

    with _lock:
        if device not in _port_state:
            return
        if probe:
            _port_state[device].update({
                "status": "connected",
                "sensor_type": probe.get("sensor_type"),
                "baud": probe.get("baud"),
                "config": probe.get("config"),
                "raw": probe.get("raw"),
                "updated": datetime.now().strftime("%H:%M:%S"),
            })
        else:
            _port_state[device].update({
                "status": "no response",
                "updated": datetime.now().strftime("%H:%M:%S"),
            })


def _bg_monitor(poll: float = 3.0):
    """Background thread: populate and keep _port_state current."""
    # Seed with ports already connected at startup
    initial = list(serial.tools.list_ports.comports())
    with _lock:
        for p in initial:
            _port_state[p.device] = _make_entry(p, "probing")

    for p in initial:
        _executor.submit(_probe_and_update, p.device)

    known = {p.device for p in initial}

    while True:
        try:
            import time
            time.sleep(poll)
            current_ports = {p.device: p for p in serial.tools.list_ports.comports()}
            current_set = set(current_ports.keys())

            removed = known - current_set
            added = current_set - known

            with _lock:
                for dev in removed:
                    if dev in _port_state:
                        _port_state[dev]["status"] = "disconnected"
                for dev in added:
                    _port_state[dev] = _make_entry(current_ports[dev], "probing")

            for dev in added:
                _executor.submit(_probe_and_update, dev)

            known = current_set

        except Exception as exc:
            print(f"[sensor_dashboard] bg monitor error: {exc}")


# ── Colours ───────────────────────────────────────────────────────────────────

SENSOR_COLORS = {
    "pH":          "#4CAF50",
    "DO":          "#2196F3",
    "Temperature": "#FF9800",
    "Foam":        "#9C27B0",
    "Stirrer":     "#00BCD4",
    "Flow":        "#F44336",
    "UNKNOWN":     "#9E9E9E",
}

STATUS_COLORS = {
    "connected":    "#4CAF50",
    "disconnected": "#F44336",
    "probing":      "#FF9800",
    "no response":  "#757575",
}


def _sensor_card(info: dict) -> html.Div:
    sensor  = info.get("sensor_type") or "UNKNOWN"
    status  = info.get("status", "unknown")
    color   = SENSOR_COLORS.get(sensor, "#9E9E9E")
    s_color = STATUS_COLORS.get(status, "#9E9E9E")
    raw     = info.get("raw") or ""
    preview = repr(raw[:100]) if raw else "—"
    device  = info["device"]

    label_row = html.Div()
    if sensor == "UNKNOWN" and raw:
        label_row = html.Div(
            style={"marginTop": "10px", "display": "flex", "gap": "8px"},
            children=[
                dcc.Input(
                    id={"type": "label-input", "device": device},
                    placeholder="Type sensor name (pH, DO, Temperature…)",
                    debounce=False,
                    style={
                        "padding": "4px 8px", "borderRadius": "4px",
                        "border": "1px solid #555", "background": "#111",
                        "color": "#fff", "width": "240px",
                    },
                ),
                html.Button(
                    "Save label",
                    id={"type": "label-btn", "device": device},
                    n_clicks=0,
                    style={
                        "background": color, "color": "#fff", "border": "none",
                        "borderRadius": "4px", "padding": "5px 14px", "cursor": "pointer",
                    },
                ),
            ],
        )

    return html.Div(
        style={
            "border": f"2px solid {color}", "borderRadius": "10px",
            "padding": "16px", "marginBottom": "14px", "background": "#1e1e2e",
        },
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                children=[
                    html.Div([
                        html.Span(sensor, style={
                            "fontSize": "20px", "fontWeight": "bold",
                            "color": color, "marginRight": "12px",
                        }),
                        html.Span(device, style={"color": "#aaa", "fontSize": "13px"}),
                    ]),
                    html.Span(status.upper(), style={
                        "background": s_color, "color": "#fff",
                        "padding": "3px 10px", "borderRadius": "12px",
                        "fontSize": "11px", "fontWeight": "bold",
                    }),
                ],
            ),
            html.Div(info.get("description") or "",
                     style={"color": "#888", "fontSize": "12px", "marginTop": "4px"}),
            html.Div(
                style={"marginTop": "10px", "display": "flex", "gap": "24px"},
                children=[
                    html.Div([html.Span("Baud: ", style={"color": "#aaa"}),
                              html.Span(str(info.get("baud") or "—"), style={"color": "#fff"})]),
                    html.Div([html.Span("Config: ", style={"color": "#aaa"}),
                              html.Span(info.get("config") or "—", style={"color": "#fff"})]),
                    html.Div([html.Span("Updated: ", style={"color": "#aaa"}),
                              html.Span(info.get("updated") or "—", style={"color": "#fff"})]),
                ],
            ),
            html.Div(
                style={"marginTop": "8px"},
                children=[
                    html.Span("Response: ", style={"color": "#aaa", "fontSize": "12px"}),
                    html.Code(preview, style={
                        "background": "#111", "color": "#0f0",
                        "padding": "2px 6px", "borderRadius": "4px",
                        "fontSize": "11px", "wordBreak": "break-all",
                    }),
                ],
            ) if raw else html.Div(),
            label_row,
        ],
    )


# ── App layout ────────────────────────────────────────────────────────────────

app = Dash(__name__, title="Sensor Detection")
app.layout = html.Div(
    style={"background": "#13131f", "minHeight": "100vh", "padding": "30px",
           "fontFamily": "monospace", "color": "#fff"},
    children=[
        html.H2("RS-232 Sensor Detection", style={"marginBottom": "4px"}),
        html.Div(id="subtitle", style={"color": "#888", "fontSize": "13px", "marginBottom": "24px"}),
        dcc.Interval(id="tick", interval=3000, n_intervals=0),
        html.Div(id="cards"),
        html.Div(id="label-feedback",
                 style={"color": "#4CAF50", "marginTop": "10px", "fontSize": "13px"}),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("cards", "children"),
    Output("subtitle", "children"),
    Input("tick", "n_intervals"),
)
def refresh(_n):
    with _lock:
        state = copy.deepcopy(_port_state)

    if not state:
        return (
            html.Div(
                "No serial ports detected. Plug in a sensor to begin.",
                style={"color": "#888", "marginTop": "40px", "textAlign": "center"},
            ),
            f"Last scan: {datetime.now().strftime('%H:%M:%S')} — waiting for devices",
        )

    cards = [_sensor_card(info) for info in state.values()]
    connected = sum(1 for v in state.values() if v.get("status") == "connected")
    subtitle = (
        f"Last scan: {datetime.now().strftime('%H:%M:%S')} — "
        f"{len(state)} port(s) found, {connected} connected"
    )
    return cards, subtitle


@app.callback(
    Output("label-feedback", "children"),
    Input({"type": "label-btn", "device": ALL}, "n_clicks"),
    State({"type": "label-input", "device": ALL}, "value"),
    prevent_initial_call=True,
)
def save_label(n_clicks_list, label_values):
    ctx = callback_context
    if not ctx.triggered:
        return no_update

    # Guard against spurious fires when a new card is rendered (n_clicks=0)
    triggered_n = ctx.triggered[0]["value"]
    if not triggered_n:
        return no_update

    prop_id = ctx.triggered[0]["prop_id"]
    try:
        id_dict = json.loads(prop_id.split(".")[0])
        device = id_dict["device"]
    except Exception:
        return no_update

    # Find the corresponding label value by matching device in the ALL state list
    states_list = ctx.states_list[0]
    idx = next((i for i, s in enumerate(states_list) if s["id"]["device"] == device), None)
    if idx is None:
        return no_update

    label_value = (label_values[idx] or "").strip() if idx < len(label_values) else ""
    if not label_value:
        return "Enter a sensor name first."

    with _lock:
        info = copy.deepcopy(_port_state.get(device, {}))

    # Auto-derive a specific pattern from the raw response so we don't save ".*"
    raw_sample = info.get("raw") or ""
    default_pat = re.escape(raw_sample[:8].strip()) if raw_sample else ".*"

    entry = {
        "device": device,
        "description": info.get("description", ""),
        "vid": info.get("vid"),
        "pid": info.get("pid"),
        "baud": info.get("baud"),
        "config": info.get("config"),
        "raw_sample": raw_sample[:200],
        "sensor_type": label_value,
        "pattern": default_pat,
    }
    save_signature(entry)

    with _lock:
        if device in _port_state:
            _port_state[device]["sensor_type"] = label_value

    return f"Saved '{label_value}' for {device} (pattern: {default_pat})"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t = threading.Thread(target=_bg_monitor, daemon=True)
    t.start()
    print("Dashboard: http://localhost:8060")
    app.run(debug=False, port=8060)
