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

sys.path.insert(0, str(Path(__file__).parent.parent))

import serial.tools.list_ports
from dash import ALL, Dash, Input, Output, State, callback_context, dcc, html, no_update

from src.serial_detector import probe_port, save_signature

# ── Shared state ──────────────────────────────────────────────────────────────
_lock = threading.Lock()
_port_state: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=4)


def _make_entry(port, status: str) -> dict:
    return {
        "device":      port.device,
        "description": port.description or "",
        "vid":         port.vid,
        "pid":         port.pid,
        "status":      status,
        "sensor_type": None,
        "baud":        None,
        "config":      None,
        "raw":         None,
        "updated":     datetime.now().strftime("%H:%M:%S"),
    }


def _probe_and_update(device: str):
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
                "status":      "connected",
                "sensor_type": probe.get("sensor_type"),
                "baud":        probe.get("baud"),
                "config":      probe.get("config"),
                "raw":         probe.get("raw"),
                "updated":     datetime.now().strftime("%H:%M:%S"),
            })
        else:
            _port_state[device].update({
                "status":  "no response",
                "updated": datetime.now().strftime("%H:%M:%S"),
            })


def _bg_monitor(poll: float = 3.0):
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
            current_set   = set(current_ports.keys())

            removed = known - current_set
            added   = current_set - known

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


# ── Design tokens ─────────────────────────────────────────────────────────────
FONT    = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
BG      = "#f8fafc"
SURFACE = "#ffffff"
BORDER  = "#e2e8f0"
TEXT    = "#0f172a"
MUTED   = "#64748b"
BLUE    = "#2563eb"

SENSOR_COLORS = {
    "pH":          "#16a34a",
    "DO":          "#2563eb",
    "Temperature": "#ea580c",
    "Foam":        "#7c3aed",
    "Stirrer":     "#0891b2",
    "Flow":        "#dc2626",
    "UNKNOWN":     "#94a3b8",
}

STATUS_COLORS = {
    "connected":    "#16a34a",
    "disconnected": "#dc2626",
    "probing":      "#d97706",
    "no response":  "#94a3b8",
}

STATUS_BG = {
    "connected":    "#f0fdf4",
    "disconnected": "#fef2f2",
    "probing":      "#fffbeb",
    "no response":  "#f8fafc",
}


def _sensor_card(info: dict) -> html.Div:
    sensor  = info.get("sensor_type") or "UNKNOWN"
    status  = info.get("status", "unknown")
    color   = SENSOR_COLORS.get(sensor, "#94a3b8")
    s_color = STATUS_COLORS.get(status, "#94a3b8")
    s_bg    = STATUS_BG.get(status, "#f8fafc")
    raw     = info.get("raw") or ""
    preview = repr(raw[:120]) if raw else None
    device  = info["device"]

    label_row = html.Div()
    if sensor == "UNKNOWN" and raw:
        label_row = html.Div(
            style={"marginTop": "12px", "display": "flex", "gap": "8px"},
            children=[
                dcc.Input(
                    id={"type": "label-input", "device": device},
                    placeholder="Sensor name (pH, DO, Temperature...)",
                    debounce=False,
                    style={
                        "padding": "5px 10px", "borderRadius": "4px",
                        "border": f"1px solid {BORDER}", "backgroundColor": BG,
                        "color": TEXT, "fontFamily": FONT, "fontSize": "0.875rem",
                        "width": "260px",
                    },
                ),
                html.Button(
                    "Save label",
                    id={"type": "label-btn", "device": device},
                    n_clicks=0,
                    style={
                        "backgroundColor": BLUE, "color": "#fff", "border": "none",
                        "borderRadius": "4px", "padding": "5px 14px",
                        "cursor": "pointer", "fontFamily": FONT,
                        "fontSize": "0.8rem", "fontWeight": "500",
                    },
                ),
            ],
        )

    return html.Div(
        style={
            "backgroundColor": SURFACE,
            "border": f"1px solid {BORDER}",
            "borderLeft": f"4px solid {color}",
            "borderRadius": "6px",
            "padding": "16px 18px",
            "marginBottom": "10px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
        },
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between",
                       "alignItems": "flex-start"},
                children=[
                    html.Div([
                        html.Span(sensor, style={
                            "fontSize": "1rem", "fontWeight": "600",
                            "color": color, "marginRight": "10px",
                            "fontFamily": FONT,
                        }),
                        html.Span(device, style={
                            "color": MUTED, "fontSize": "0.8rem",
                            "fontFamily": FONT,
                        }),
                    ]),
                    html.Span(status.upper(), style={
                        "backgroundColor": s_bg, "color": s_color,
                        "border": f"1px solid {s_color}",
                        "padding": "2px 10px", "borderRadius": "12px",
                        "fontSize": "0.7rem", "fontWeight": "600",
                        "fontFamily": FONT, "letterSpacing": "0.04em",
                        "whiteSpace": "nowrap",
                    }),
                ],
            ),

            html.Div(info.get("description") or "",
                     style={"color": MUTED, "fontSize": "0.8rem",
                            "fontFamily": FONT, "marginTop": "4px"}),

            html.Div(
                style={"marginTop": "10px", "display": "flex", "gap": "24px"},
                children=[
                    _kv("Baud",    str(info.get("baud") or "—")),
                    _kv("Config",  info.get("config") or "—"),
                    _kv("Updated", info.get("updated") or "—"),
                ],
            ),

            html.Div(
                style={"marginTop": "8px"},
                children=[
                    html.Span("Response  ", style={
                        "color": MUTED, "fontSize": "0.75rem",
                        "fontFamily": FONT, "fontWeight": "600",
                        "textTransform": "uppercase", "letterSpacing": "0.06em",
                    }),
                    html.Code(preview, style={
                        "backgroundColor": "#f1f5f9", "color": TEXT,
                        "padding": "2px 8px", "borderRadius": "4px",
                        "fontSize": "0.8rem", "fontFamily": "monospace",
                        "wordBreak": "break-all",
                    }),
                ],
            ) if preview else html.Div(),

            label_row,
        ],
    )


def _kv(label, value):
    return html.Div([
        html.Span(f"{label}  ", style={
            "color": MUTED, "fontSize": "0.75rem", "fontFamily": FONT,
            "fontWeight": "600", "textTransform": "uppercase",
            "letterSpacing": "0.05em",
        }),
        html.Span(value, style={
            "color": TEXT, "fontSize": "0.875rem", "fontFamily": FONT,
        }),
    ])


# ── App layout ────────────────────────────────────────────────────────────────
app = Dash(__name__, title="Sensor Detection")

app.layout = html.Div(
    style={"backgroundColor": BG, "minHeight": "100vh",
           "fontFamily": FONT, "color": TEXT},
    children=[
        html.Div([
            html.Div([
                html.H1("RS-232 Sensor Detection",
                        style={"margin": 0, "fontSize": "1.1rem",
                               "fontWeight": "600", "color": TEXT,
                               "fontFamily": FONT}),
                html.Div(id="subtitle",
                         style={"color": MUTED, "fontSize": "0.8rem",
                                "fontFamily": FONT, "marginTop": "2px"}),
            ]),
        ], style={
            "padding": "14px 28px", "backgroundColor": SURFACE,
            "borderBottom": f"1px solid {BORDER}",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
        }),

        html.Div([
            dcc.Interval(id="tick", interval=3000, n_intervals=0),
            html.Div(id="cards"),
            html.Div(id="label-feedback",
                     style={"color": "#16a34a", "marginTop": "8px",
                            "fontSize": "0.875rem", "fontFamily": FONT}),
        ], style={"padding": "20px 28px", "maxWidth": "860px"}),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("cards",    "children"),
    Output("subtitle", "children"),
    Input("tick",      "n_intervals"),
)
def refresh(_n):
    with _lock:
        state = copy.deepcopy(_port_state)

    if not state:
        return (
            html.Div(
                "No serial ports detected. Plug in a sensor to begin.",
                style={"color": MUTED, "marginTop": "40px",
                       "textAlign": "center", "fontFamily": FONT},
            ),
            f"Last scan: {datetime.now().strftime('%H:%M:%S')}  —  waiting for devices",
        )

    cards     = [_sensor_card(info) for info in state.values()]
    connected = sum(1 for v in state.values() if v.get("status") == "connected")
    subtitle  = (
        f"Last scan: {datetime.now().strftime('%H:%M:%S')}  —  "
        f"{len(state)} port(s) found, {connected} connected"
    )
    return cards, subtitle


@app.callback(
    Output("label-feedback", "children"),
    Input({"type": "label-btn",   "device": ALL}, "n_clicks"),
    State({"type": "label-input", "device": ALL}, "value"),
    prevent_initial_call=True,
)
def save_label(n_clicks_list, label_values):
    ctx = callback_context
    if not ctx.triggered:
        return no_update

    triggered_n = ctx.triggered[0]["value"]
    if not triggered_n:
        return no_update

    prop_id = ctx.triggered[0]["prop_id"]
    try:
        id_dict = json.loads(prop_id.split(".")[0])
        device  = id_dict["device"]
    except Exception:
        return no_update

    states_list = ctx.states_list[0]
    idx = next((i for i, s in enumerate(states_list)
                if s["id"]["device"] == device), None)
    if idx is None:
        return no_update

    label_value = (label_values[idx] or "").strip() if idx < len(label_values) else ""
    if not label_value:
        return "Enter a sensor name first."

    with _lock:
        info = copy.deepcopy(_port_state.get(device, {}))

    raw_sample  = info.get("raw") or ""
    default_pat = re.escape(raw_sample[:8].strip()) if raw_sample else ".*"

    entry = {
        "device":      device,
        "description": info.get("description", ""),
        "vid":         info.get("vid"),
        "pid":         info.get("pid"),
        "baud":        info.get("baud"),
        "config":      info.get("config"),
        "raw_sample":  raw_sample[:200],
        "sensor_type": label_value,
        "pattern":     default_pat,
    }
    save_signature(entry)

    with _lock:
        if device in _port_state:
            _port_state[device]["sensor_type"] = label_value

    return f"Saved '{label_value}' for {device}  (pattern: {default_pat})"


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=_bg_monitor, daemon=True)
    t.start()
    print("Dashboard: http://localhost:8060")
    app.run(debug=False, port=8060)
