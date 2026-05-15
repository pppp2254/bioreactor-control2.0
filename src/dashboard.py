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

# ── Load data ─────────────────────────────────────────────────────────────────
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

COLORS = {
    "result-batch_5": "#1a56db",   # blue
    "result-batch_7": "#e67e22",   # orange
}

FONT = "Times New Roman"
METHANOL_PULSE_FREQ = 0.714   # cycles/h — matches transforms.py

LAYOUT_BASE = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family=FONT, color="#111111", size=12),
    legend=dict(
        bgcolor="#ffffff",
        bordercolor="#cccccc",
        borderwidth=1,
        font=dict(family=FONT, size=11),
    ),
    xaxis=dict(
        showgrid=True, gridcolor="#e5e5e5", gridwidth=1,
        linecolor="#111111", linewidth=1,
        tickfont=dict(family=FONT),
        title_font=dict(family=FONT),
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#e5e5e5", gridwidth=1,
        linecolor="#111111", linewidth=1,
        tickfont=dict(family=FONT),
        title_font=dict(family=FONT),
    ),
    margin=dict(l=60, r=20, t=50, b=50),
)

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.title = "Bioreactor Signal Dashboard"

LABEL_STYLE = {
    "fontFamily": FONT,
    "fontSize": "0.7rem",
    "textTransform": "uppercase",
    "letterSpacing": "0.08em",
    "color": "#666",
    "marginBottom": "6px",
    "marginTop": "16px",
    "display": "block",
}
CHECK_STYLE  = {"fontFamily": FONT, "fontSize": "0.9rem", "color": "#111"}
INPUT_STYLE  = {"marginRight": "6px", "accentColor": "#1a56db"}

app.layout = html.Div([

    # Header
    html.Div([
        html.H1("Bioreactor Signal Dashboard",
                style={"margin": 0, "fontSize": "1.3rem",
                       "fontFamily": FONT, "fontWeight": "normal",
                       "color": "#111", "letterSpacing": "0.02em"}),
        html.P("HPV52 L1 · Hansenula polymorpha · Signal Processing",
               style={"margin": "2px 0 0", "fontSize": "0.82rem",
                      "fontFamily": FONT, "color": "#666"}),
    ], style={
        "padding": "16px 28px",
        "borderBottom": "2px solid #111",
        "backgroundColor": "#fff",
    }),

    # Tab bar
    dcc.Tabs(id="main-tabs", value="signals", children=[
        dcc.Tab(label="Signals",      value="signals"),
        dcc.Tab(label="FFT Spectrum", value="fft"),
        dcc.Tab(label="STFT",         value="stft"),
    ], style={"fontFamily": FONT, "fontSize": "0.9rem",
              "borderBottom": "1px solid #ddd"}),

    html.Div(id="tab-body",
             style={"display": "flex", "height": "calc(100vh - 110px)",
                    "overflow": "hidden"}),

], style={"fontFamily": FONT, "backgroundColor": "#fff",
          "height": "100vh", "overflow": "hidden"})


# ── Tab router ────────────────────────────────────────────────────────────────
@app.callback(Output("tab-body", "children"), Input("main-tabs", "value"))
def render_tab(tab):

    sidebar_base = html.Div([
        html.Span("Batch", style=LABEL_STYLE),
        dcc.Checklist(
            id="batch-select",
            options=[{"label": f"  {k.replace('result-','').replace('_',' ').title()}",
                      "value": k} for k in batches],
            value=list(batches.keys()),
            style=CHECK_STYLE,
            inputStyle=INPUT_STYLE,
        ),
    ], style={
        "width": "210px", "flexShrink": "0",
        "padding": "16px 20px",
        "borderRight": "1px solid #ddd",
        "backgroundColor": "#fafafa",
        "overflowY": "auto",
    })

    if tab == "signals":
        sidebar = html.Div([
            html.Span("Batch", style=LABEL_STYLE),
            dcc.Checklist(
                id="batch-select",
                options=[{"label": f"  {k.replace('result-','').replace('_',' ').title()}",
                          "value": k} for k in batches],
                value=list(batches.keys()),
                style=CHECK_STYLE,
                inputStyle=INPUT_STYLE,
            ),
            html.Span("Signals", style=LABEL_STYLE),
            dcc.Checklist(
                id="signal-select",
                options=[{"label": f"  {v}", "value": k}
                         for k, v in SIGNALS.items()],
                value=["DO_pct", "pH", "OD660"],
                style=CHECK_STYLE,
                inputStyle=INPUT_STYLE,
            ),
            html.Span("Layout", style=LABEL_STYLE),
            dcc.RadioItems(
                id="view-mode",
                options=[
                    {"label": "  Stack signals",   "value": "stack"},
                    {"label": "  Overlay batches", "value": "overlay"},
                ],
                value="stack",
                style=CHECK_STYLE,
                inputStyle=INPUT_STYLE,
            ),
        ], style={
            "width": "210px", "flexShrink": "0",
            "padding": "16px 20px",
            "borderRight": "1px solid #ddd",
            "backgroundColor": "#fafafa",
            "overflowY": "auto",
        })
        chart_area = html.Div([
            dcc.Graph(id="main-chart",
                      style={"height": "88vh"},
                      config={"displayModeBar": True,
                              "toImageButtonOptions":
                              {"format": "svg", "filename": "bioreactor_signals"}}),
        ], style={"flex": "1", "overflowY": "auto"})
        return [sidebar, chart_area]

    elif tab == "fft":
        sidebar = html.Div([
            html.Span("Batch", style=LABEL_STYLE),
            dcc.Checklist(
                id="batch-select",
                options=[{"label": f"  {k.replace('result-','').replace('_',' ').title()}",
                          "value": k} for k in batches],
                value=list(batches.keys()),
                style=CHECK_STYLE,
                inputStyle=INPUT_STYLE,
            ),
            html.Span("Signal", style=LABEL_STYLE),
            dcc.Dropdown(
                id="fft-signal",
                options=[{"label": v, "value": k} for k, v in SIGNALS.items()],
                value="DO_pct",
                clearable=False,
                style={"fontFamily": FONT, "fontSize": "0.88rem"},
            ),
        ], style={
            "width": "210px", "flexShrink": "0",
            "padding": "16px 20px",
            "borderRight": "1px solid #ddd",
            "backgroundColor": "#fafafa",
            "overflowY": "auto",
        })
        chart_area = html.Div([
            dcc.Graph(id="fft-chart",
                      style={"height": "88vh"},
                      config={"displayModeBar": False}),
        ], style={"flex": "1", "overflowY": "auto"})
        return [sidebar, chart_area]

    elif tab == "stft":
        sidebar = html.Div([
            html.Span("Batch", style=LABEL_STYLE),
            dcc.RadioItems(
                id="stft-batch",
                options=[{"label": f"  {k.replace('result-','').replace('_',' ').title()}",
                          "value": k} for k in batches],
                value=list(batches.keys())[0],
                style=CHECK_STYLE,
                inputStyle=INPUT_STYLE,
            ),
            html.Span("Signal", style=LABEL_STYLE),
            dcc.Dropdown(
                id="stft-signal",
                options=[{"label": v, "value": k} for k, v in SIGNALS.items()
                         if k in ("DO_pct", "pH", "DO_rescaled", "stirrer")],
                value="DO_pct",
                clearable=False,
                style={"fontFamily": FONT, "fontSize": "0.88rem"},
            ),
            html.Span("Window (h)", style=LABEL_STYLE),
            dcc.Slider(id="stft-window", min=2, max=8, step=1, value=4,
                       marks={i: str(i) for i in range(2, 9)},
                       tooltip={"placement": "bottom"}),
            html.Span("Step (h)", style=LABEL_STYLE),
            dcc.Slider(id="stft-step", min=0.25, max=2, step=0.25, value=0.5,
                       marks={0.25: "0.25", 1: "1", 2: "2"},
                       tooltip={"placement": "bottom"}),
            html.Hr(style={"borderColor": "#ddd", "marginTop": "20px"}),
            html.Div(id="stft-info",
                     style={"fontFamily": FONT, "fontSize": "0.82rem",
                            "color": "#666", "lineHeight": "1.6"}),
        ], style={
            "width": "210px", "flexShrink": "0",
            "padding": "16px 20px",
            "borderRight": "1px solid #ddd",
            "backgroundColor": "#fafafa",
            "overflowY": "auto",
        })
        chart_area = html.Div([
            dcc.Graph(id="stft-chart",
                      style={"height": "88vh"},
                      config={"displayModeBar": True,
                              "toImageButtonOptions":
                              {"format": "svg", "filename": "stft_heatmap"}}),
        ], style={"flex": "1", "overflowY": "auto"})
        return [sidebar, chart_area]

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
                    line=dict(color=COLORS.get(batch, "#333"), width=1.5),
                    showlegend=(row == 1),
                ), row=row, col=1)

        for i in range(1, n_sigs + 1):
            fig.update_xaxes(
                showgrid=True, gridcolor="#e5e5e5",
                linecolor="#111", tickfont=dict(family=FONT),
                row=i, col=1
            )
            fig.update_yaxes(
                showgrid=True, gridcolor="#e5e5e5",
                linecolor="#111", tickfont=dict(family=FONT),
                title_font=dict(family=FONT),
                row=i, col=1
            )
        fig.update_annotations(font=dict(family=FONT, size=11, color="#333"))
        fig.update_layout(
            height=max(300, n_sigs * 190),
            **{k: v for k, v in LAYOUT_BASE.items()
               if k not in ("xaxis", "yaxis")},
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
                    name=f"{batch.replace('result-','').replace('_',' ').title()} · {SIGNALS[sig]}",
                    line=dict(color=COLORS.get(batch, "#333"), width=1.5),
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
        color = COLORS.get(batch, "#333")
        fill_rgba = ("rgba(26,86,219,0.08)" if batch == "result-batch_5"
                     else "rgba(230,126,34,0.08)")
        fig.add_trace(go.Scatter(
            x=freqs[1:], y=amps[1:],
            name=batch.replace("result-", "").replace("_", " ").title(),
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor=fill_rgba,
        ))

    # Mark methanol pulse frequency
    fig.add_vline(x=METHANOL_PULSE_FREQ, line_dash="dash",
                  line_color="#dc2626", line_width=1,
                  annotation_text="MeOH pulse (0.71/h)",
                  annotation_font=dict(family=FONT, size=10, color="#dc2626"),
                  annotation_position="top right")

    fig.update_layout(
        **LAYOUT_BASE,
        xaxis_title="Frequency (cycles/h)",
        yaxis_title="Amplitude",
        title=dict(text=f"FFT Spectrum — {SIGNALS.get(sig, sig)}",
                   font=dict(family=FONT, size=12, color="#333")),
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

    times  = np.array(result["stft_times"])       # shape: [n_windows]
    freqs  = np.array(result["stft_freqs"])        # shape: [n_freqs]
    power  = np.array(result["stft_power"])        # shape: [n_windows, n_freqs]
    pulse  = np.array(result["pulse_power"])       # shape: [n_windows]

    # Clip frequency axis to a readable range (0–3 cycles/h covers methanol pulse)
    freq_max = 3.0
    freq_mask = freqs <= freq_max
    freqs_plot = freqs[freq_mask]
    power_plot = power[:, freq_mask]               # [n_windows, n_freqs_clipped]

    # Log scale for better contrast (add small epsilon to avoid log(0))
    power_log = np.log10(power_plot + 1e-12)

    batch_label = batch.replace("result-", "").replace("_", " ").title()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f"STFT Power Heatmap — {SIGNALS.get(sig, sig)} ({batch_label})",
            f"Power at Methanol Pulse Freq ({METHANOL_PULSE_FREQ:.2f} cycles/h)",
        ],
        vertical_spacing=0.12,
        row_heights=[0.65, 0.35],
    )

    # ── Heatmap ───────────────────────────────────────────────────────────────
    fig.add_trace(go.Heatmap(
        x=times,
        y=freqs_plot,
        z=power_log.T,                             # transpose: [n_freqs, n_windows]
        colorscale="Viridis",
        colorbar=dict(
            title=dict(text="log₁₀ Power", font=dict(family=FONT, size=11)),
            thickness=14,
            len=0.6,
            y=0.7,
            tickfont=dict(family=FONT, size=10),
        ),
        hovertemplate="Time: %{x:.1f} h<br>Freq: %{y:.3f} /h<br>log Power: %{z:.2f}<extra></extra>",
    ), row=1, col=1)

    # Mark methanol pulse frequency line on heatmap
    fig.add_hline(y=METHANOL_PULSE_FREQ, line_dash="dash",
                  line_color="#ef4444", line_width=1.5,
                  annotation_text=f"MeOH {METHANOL_PULSE_FREQ}/h",
                  annotation_font=dict(family=FONT, size=10, color="#ef4444"),
                  annotation_position="right",
                  row=1, col=1)

    # ── Pulse-power time series ───────────────────────────────────────────────
    color = COLORS.get(batch, "#1a56db")
    fig.add_trace(go.Scatter(
        x=times, y=pulse,
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=("rgba(26,86,219,0.12)" if batch == "result-batch_5"
                   else "rgba(230,126,34,0.12)"),
        name="Pulse power",
        hovertemplate="Time: %{x:.1f} h<br>Power: %{y:.4f}<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
        showlegend=False,
        height=700,
    )
    fig.update_xaxes(title_text="Time (h)", tickfont=dict(family=FONT),
                     gridcolor="#e5e5e5", row=2, col=1)
    fig.update_xaxes(tickfont=dict(family=FONT), gridcolor="#e5e5e5",
                     row=1, col=1)
    fig.update_yaxes(title_text="Frequency (cycles/h)",
                     tickfont=dict(family=FONT), gridcolor="#e5e5e5",
                     row=1, col=1)
    fig.update_yaxes(title_text="Power",
                     tickfont=dict(family=FONT), gridcolor="#e5e5e5",
                     row=2, col=1)

    # Info panel
    pulse_onset = None
    if len(pulse) > 0:
        threshold = pulse.max() * 0.2
        above = np.where(pulse > threshold)[0]
        if len(above):
            pulse_onset = times[above[0]]

    info = [
        html.B(f"{len(times)} windows"),
        html.Br(),
        f"Window: {window_h} h · Step: {step_h} h",
        html.Br(),
        f"Freq resolution: {freqs[1] - freqs[0]:.3f} /h" if len(freqs) > 1 else "",
        html.Br(),
        f"Pulse freq: {METHANOL_PULSE_FREQ} /h",
        html.Br(),
        html.Br(),
        html.B("Max pulse power: "),
        f"{pulse.max():.4f}",
        html.Br(),
        html.B("Onset (>20% max): "),
        f"{pulse_onset:.1f} h" if pulse_onset is not None else "—",
    ]

    return fig, info


if __name__ == "__main__":
    print("Starting at http://127.0.0.1:8050")
    app.run(debug=True)
