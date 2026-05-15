import pandas as pd
import numpy as np
from pathlib import Path
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
sys.path.insert(0, "src")

from transforms import rolling_stft

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("outputs/processed_data")
batches = {}
for f in sorted(DATA_DIR.glob("*.csv")):
    batches[f.stem] = pd.read_csv(f)

SIGNALS = {
    "DO_pct":        "DO (%)",
    "DO_rescaled":   "DO Rescaled (%)",
    "pH":            "pH",
    "temp_c":        "Temperature (°C)",
    "stirrer":       "Stirrer (rpm)",
    "OD660":         "OD660 (Biomass)",
    "glycerol_gL":   "Glycerol (g/L)",
    "methanol_gL":   "Methanol (g/L)",
    "DCW_gL":        "DCW (g/L)",
    "L1_yield_mgL":  "L1 Yield (mg/L)",
}

METHANOL_PULSE_FREQ = 0.714

# ── Design tokens ─────────────────────────────────────────────────────────────
FONT    = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
BG      = "#f8fafc"
SURFACE = "#ffffff"
BORDER  = "#e2e8f0"
TEXT    = "#0f172a"
MUTED   = "#64748b"
BLUE    = "#2563eb"
ORANGE  = "#ea580c"
RED     = "#dc2626"

BATCH_COLORS = {
    "result-batch_5": BLUE,
    "result-batch_7": ORANGE,
}

LAYOUT_BASE = dict(
    template="plotly_white",
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family=FONT, color=TEXT, size=12),
    legend=dict(bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1,
                font=dict(family=FONT, size=11)),
    xaxis=dict(showgrid=True, gridcolor="#f1f5f9", gridwidth=1,
               linecolor=BORDER, tickfont=dict(family=FONT),
               title_font=dict(family=FONT)),
    yaxis=dict(showgrid=True, gridcolor="#f1f5f9", gridwidth=1,
               linecolor=BORDER, tickfont=dict(family=FONT),
               title_font=dict(family=FONT)),
    margin=dict(l=60, r=20, t=40, b=50),
)

LABEL_S = {
    "fontFamily": FONT, "fontSize": "0.7rem", "textTransform": "uppercase",
    "letterSpacing": "0.08em", "color": MUTED, "fontWeight": "600",
    "marginBottom": "6px", "marginTop": "16px", "display": "block",
}
CHECK_S = {"fontFamily": FONT, "fontSize": "0.875rem", "color": TEXT}
INPUT_S = {"marginRight": "8px", "accentColor": BLUE}
SIDEBAR_S = {
    "width": "220px", "flexShrink": "0", "padding": "20px",
    "borderRight": f"1px solid {BORDER}", "backgroundColor": SURFACE,
    "overflowY": "auto",
}
TAB_S = {
    "padding": "8px 20px", "fontFamily": FONT, "fontSize": "0.875rem",
    "color": MUTED, "backgroundColor": BG, "border": "none",
    "borderBottom": "2px solid transparent",
}
TAB_SEL = {
    "padding": "8px 20px", "fontFamily": FONT, "fontSize": "0.875rem",
    "color": BLUE, "fontWeight": "500", "backgroundColor": BG,
    "borderTop": "none", "borderLeft": "none", "borderRight": "none",
    "borderBottom": f"2px solid {BLUE}",
}

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.title = "Bioreactor Signal Dashboard"

app.layout = html.Div([
    html.Div([
        html.H1("Bioreactor Signal Dashboard",
                style={"margin": 0, "fontSize": "1.1rem", "fontFamily": FONT,
                       "fontWeight": "600", "color": TEXT}),
        html.P("HPV52 L1  ·  Hansenula polymorpha  ·  Signal Processing",
               style={"margin": "2px 0 0", "fontSize": "0.8rem",
                      "fontFamily": FONT, "color": MUTED}),
    ], style={"padding": "14px 28px", "borderBottom": f"1px solid {BORDER}",
              "backgroundColor": SURFACE,
              "boxShadow": "0 1px 3px rgba(0,0,0,0.06)"}),

    dcc.Tabs(id="main-tabs", value="signals", children=[
        dcc.Tab(label="Signals",      value="signals",
                style=TAB_S, selected_style=TAB_SEL),
        dcc.Tab(label="FFT Spectrum", value="fft",
                style=TAB_S, selected_style=TAB_SEL),
        dcc.Tab(label="STFT",         value="stft",
                style=TAB_S, selected_style=TAB_SEL),
    ], style={"fontFamily": FONT, "backgroundColor": BG,
              "borderBottom": f"1px solid {BORDER}"}),

    html.Div(id="tab-body",
             style={"display": "flex", "height": "calc(100vh - 106px)",
                    "overflow": "hidden", "backgroundColor": BG}),

], style={"fontFamily": FONT, "backgroundColor": BG,
          "height": "100vh", "overflow": "hidden"})


def _chart_area(graph_id, height="88vh", config=None):
    cfg = config or {"displayModeBar": True}
    return html.Div([
        dcc.Graph(id=graph_id, style={"height": height}, config=cfg),
    ], style={"flex": "1", "overflowY": "auto", "padding": "16px 20px",
              "backgroundColor": BG})


# ── Tab router ────────────────────────────────────────────────────────────────
@app.callback(Output("tab-body", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    batch_opts = [
        {"label": f"  {k.replace('result-', '').replace('_', ' ').title()}", "value": k}
        for k in batches
    ]

    if tab == "signals":
        sidebar = html.Div([
            html.Span("Batch", style=LABEL_S),
            dcc.Checklist(id="batch-select", options=batch_opts,
                          value=list(batches.keys()),
                          style=CHECK_S, inputStyle=INPUT_S),
            html.Span("Signals", style=LABEL_S),
            dcc.Checklist(
                id="signal-select",
                options=[{"label": f"  {v}", "value": k} for k, v in SIGNALS.items()],
                value=["DO_pct", "pH", "OD660"],
                style=CHECK_S, inputStyle=INPUT_S,
            ),
            html.Span("Layout", style=LABEL_S),
            dcc.RadioItems(
                id="view-mode",
                options=[{"label": "  Stack signals",   "value": "stack"},
                         {"label": "  Overlay batches", "value": "overlay"}],
                value="stack",
                style=CHECK_S, inputStyle=INPUT_S,
            ),
        ], style=SIDEBAR_S)
        return [sidebar, _chart_area("main-chart", config={
            "displayModeBar": True,
            "toImageButtonOptions": {"format": "svg", "filename": "bioreactor_signals"},
        })]

    elif tab == "fft":
        sidebar = html.Div([
            html.Span("Batch", style=LABEL_S),
            dcc.Checklist(id="batch-select", options=batch_opts,
                          value=list(batches.keys()),
                          style=CHECK_S, inputStyle=INPUT_S),
            html.Span("Signal", style=LABEL_S),
            dcc.Dropdown(
                id="fft-signal",
                options=[{"label": v, "value": k} for k, v in SIGNALS.items()],
                value="DO_pct", clearable=False,
                style={"fontFamily": FONT, "fontSize": "0.875rem"},
            ),
        ], style=SIDEBAR_S)
        return [sidebar, _chart_area("fft-chart", config={"displayModeBar": False})]

    elif tab == "stft":
        sidebar = html.Div([
            html.Span("Batch", style=LABEL_S),
            dcc.RadioItems(id="stft-batch", options=batch_opts,
                           value=list(batches.keys())[0],
                           style=CHECK_S, inputStyle=INPUT_S),
            html.Span("Signal", style=LABEL_S),
            dcc.Dropdown(
                id="stft-signal",
                options=[{"label": v, "value": k} for k, v in SIGNALS.items()
                         if k in ("DO_pct", "pH", "DO_rescaled", "stirrer")],
                value="DO_pct", clearable=False,
                style={"fontFamily": FONT, "fontSize": "0.875rem"},
            ),
            html.Span("Window (h)", style=LABEL_S),
            dcc.Slider(id="stft-window", min=2, max=8, step=1, value=4,
                       marks={i: str(i) for i in range(2, 9)},
                       tooltip={"placement": "bottom"}),
            html.Span("Step (h)", style=LABEL_S),
            dcc.Slider(id="stft-step", min=0.25, max=2, step=0.25, value=0.5,
                       marks={0.25: "0.25", 1: "1", 2: "2"},
                       tooltip={"placement": "bottom"}),
            html.Hr(style={"borderColor": BORDER, "marginTop": "20px"}),
            html.Div(id="stft-info",
                     style={"fontFamily": FONT, "fontSize": "0.8rem",
                            "color": MUTED, "lineHeight": "1.7"}),
        ], style=SIDEBAR_S)
        return [sidebar, _chart_area("stft-chart", config={
            "displayModeBar": True,
            "toImageButtonOptions": {"format": "svg", "filename": "stft_heatmap"},
        })]

    return [html.Div("Unknown tab")]


# ── Signals callback ──────────────────────────────────────────────────────────
@app.callback(
    Output("main-chart", "figure"),
    Input("batch-select", "value"),
    Input("signal-select", "value"),
    Input("view-mode", "value"),
)
def update_main(selected_batches, selected_signals, view_mode):
    if not selected_batches or not selected_signals:
        return go.Figure()

    signals = [s for s in selected_signals if s in SIGNALS]
    n_sigs  = len(signals)

    if view_mode == "stack":
        fig = make_subplots(
            rows=n_sigs, cols=1,
            shared_xaxes=True,
            subplot_titles=[SIGNALS[s] for s in signals],
            vertical_spacing=0.08,
        )
        for row, sig in enumerate(signals, 1):
            for batch in selected_batches:
                df = batches[batch]
                if sig not in df.columns:
                    continue
                s = df[sig].dropna()
                if s.empty:
                    continue
                t = df.loc[s.index, "time_h"]
                fig.add_trace(go.Scatter(
                    x=t, y=s,
                    name=batch.replace("result-", "").replace("_", " ").title(),
                    line=dict(color=BATCH_COLORS.get(batch, TEXT), width=1.5),
                    showlegend=(row == 1),
                ), row=row, col=1)

        for i in range(1, n_sigs + 1):
            fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9",
                             linecolor=BORDER, tickfont=dict(family=FONT),
                             row=i, col=1)
            fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9",
                             linecolor=BORDER, tickfont=dict(family=FONT),
                             title_font=dict(family=FONT), row=i, col=1)
        fig.update_annotations(font=dict(family=FONT, size=11, color=MUTED))
        fig.update_layout(
            height=max(300, n_sigs * 190),
            **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
            xaxis_title="Time (h)",
        )

    else:
        fig = go.Figure()
        for sig in signals:
            for batch in selected_batches:
                df = batches[batch]
                if sig not in df.columns:
                    continue
                s = df[sig].dropna()
                if s.empty:
                    continue
                t = df.loc[s.index, "time_h"]
                fig.add_trace(go.Scatter(
                    x=t, y=s,
                    name=f"{batch.replace('result-','').replace('_',' ').title()}  {SIGNALS[sig]}",
                    line=dict(color=BATCH_COLORS.get(batch, TEXT), width=1.5),
                ))
        fig.update_layout(**LAYOUT_BASE, xaxis_title="Time (h)")

    return fig


# ── FFT callback ──────────────────────────────────────────────────────────────
@app.callback(
    Output("fft-chart", "figure"),
    Input("batch-select", "value"),
    Input("fft-signal", "value"),
)
def update_fft(selected_batches, sig):
    fig = go.Figure()
    if not selected_batches or not sig:
        return fig

    for batch in selected_batches:
        df = batches[batch]
        if sig not in df.columns:
            continue
        series = df[sig].interpolate(method="linear").fillna(0)
        if series.isna().all():
            continue
        n     = len(series)
        dt    = 1 / 12
        freqs = np.fft.rfftfreq(n, d=dt)
        amps  = np.abs(np.fft.rfft(series.values)) * 2 / n
        color = BATCH_COLORS.get(batch, TEXT)
        fill_rgba = (
            "rgba(37,99,235,0.08)" if batch == "result-batch_5"
            else "rgba(234,88,12,0.08)"
        )
        fig.add_trace(go.Scatter(
            x=freqs[1:], y=amps[1:],
            name=batch.replace("result-", "").replace("_", " ").title(),
            line=dict(color=color, width=1.5),
            fill="tozeroy", fillcolor=fill_rgba,
        ))

    fig.add_vline(x=METHANOL_PULSE_FREQ, line_dash="dash",
                  line_color=RED, line_width=1,
                  annotation_text="MeOH pulse (0.71/h)",
                  annotation_font=dict(family=FONT, size=10, color=RED),
                  annotation_position="top right")

    fig.update_layout(
        **LAYOUT_BASE,
        xaxis_title="Frequency (cycles/h)",
        yaxis_title="Amplitude",
        title=dict(text=f"FFT Spectrum  —  {SIGNALS.get(sig, sig)}",
                   font=dict(family=FONT, size=12, color=MUTED)),
    )
    return fig


# ── STFT callback ─────────────────────────────────────────────────────────────
@app.callback(
    Output("stft-chart", "figure"),
    Output("stft-info",  "children"),
    Input("stft-batch",  "value"),
    Input("stft-signal", "value"),
    Input("stft-window", "value"),
    Input("stft-step",   "value"),
)
def update_stft(batch, sig, window_h, step_h):
    empty_fig = go.Figure()
    empty_fig.update_layout(**LAYOUT_BASE)

    if not batch or not sig:
        return empty_fig, ""

    df = batches.get(batch)
    if df is None or sig not in df.columns:
        return empty_fig, f"Signal '{sig}' not in {batch}"

    series = df[sig].interpolate(method="linear").bfill().fillna(0)
    time_h = df["time_h"]
    series.name = sig

    result = rolling_stft(series, time_h, window_h=float(window_h),
                          step_h=float(step_h))

    if result["stft_times"] is None:
        return empty_fig, result.get("note", "STFT failed")

    times  = np.array(result["stft_times"])
    freqs  = np.array(result["stft_freqs"])
    power  = np.array(result["stft_power"])
    pulse  = np.array(result["pulse_power"])

    freq_mask   = freqs <= 3.0
    freqs_plot  = freqs[freq_mask]
    power_log   = np.log10(power[:, freq_mask] + 1e-12)

    batch_label = batch.replace("result-", "").replace("_", " ").title()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f"STFT Power Heatmap  —  {SIGNALS.get(sig, sig)} ({batch_label})",
            f"Power at Methanol Pulse Freq ({METHANOL_PULSE_FREQ:.2f} cycles/h)",
        ],
        vertical_spacing=0.12,
        row_heights=[0.65, 0.35],
    )

    fig.add_trace(go.Heatmap(
        x=times, y=freqs_plot, z=power_log.T,
        colorscale="Viridis",
        colorbar=dict(
            title=dict(text="log₁₀ Power", font=dict(family=FONT, size=11)),
            thickness=14, len=0.6, y=0.7,
            tickfont=dict(family=FONT, size=10),
        ),
        hovertemplate="Time: %{x:.1f} h<br>Freq: %{y:.3f} /h<br>log Power: %{z:.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_hline(y=METHANOL_PULSE_FREQ, line_dash="dash",
                  line_color=RED, line_width=1.5,
                  annotation_text=f"MeOH {METHANOL_PULSE_FREQ}/h",
                  annotation_font=dict(family=FONT, size=10, color=RED),
                  annotation_position="right", row=1, col=1)

    color = BATCH_COLORS.get(batch, BLUE)
    fill_rgba = (
        "rgba(37,99,235,0.12)" if batch == "result-batch_5"
        else "rgba(234,88,12,0.12)"
    )
    fig.add_trace(go.Scatter(
        x=times, y=pulse,
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=fill_rgba,
        name="Pulse power",
        hovertemplate="Time: %{x:.1f} h<br>Power: %{y:.4f}<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
        showlegend=False, height=700,
    )
    fig.update_xaxes(title_text="Time (h)", tickfont=dict(family=FONT),
                     gridcolor="#f1f5f9", row=2, col=1)
    fig.update_xaxes(tickfont=dict(family=FONT), gridcolor="#f1f5f9", row=1, col=1)
    fig.update_yaxes(title_text="Frequency (cycles/h)",
                     tickfont=dict(family=FONT), gridcolor="#f1f5f9", row=1, col=1)
    fig.update_yaxes(title_text="Power",
                     tickfont=dict(family=FONT), gridcolor="#f1f5f9", row=2, col=1)

    pulse_onset = None
    if len(pulse) > 0:
        above = np.where(pulse > pulse.max() * 0.2)[0]
        if len(above):
            pulse_onset = times[above[0]]

    info = [
        html.B(f"{len(times)} windows"),
        html.Br(),
        f"Window: {window_h} h  ·  Step: {step_h} h",
        html.Br(),
        f"Freq resolution: {freqs[1] - freqs[0]:.3f} /h" if len(freqs) > 1 else "",
        html.Br(),
        f"Pulse freq: {METHANOL_PULSE_FREQ} /h",
        html.Br(), html.Br(),
        html.B("Max pulse power: "), f"{pulse.max():.4f}",
        html.Br(),
        html.B("Onset (>20% max): "),
        f"{pulse_onset:.1f} h" if pulse_onset is not None else "—",
    ]

    return fig, info


if __name__ == "__main__":
    print("Starting at http://127.0.0.1:8050")
    app.run(debug=True)
