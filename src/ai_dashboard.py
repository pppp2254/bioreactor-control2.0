import pandas as pd
import numpy as np
from pathlib import Path
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
sys.path.insert(0, "src")

# ── Design tokens ─────────────────────────────────────────────────────────────
FONT    = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
BG      = "#f8fafc"
SURFACE = "#ffffff"
BORDER  = "#e2e8f0"
TEXT    = "#0f172a"
MUTED   = "#64748b"
BLUE    = "#2563eb"
GREEN   = "#16a34a"
RED     = "#dc2626"
ORANGE  = "#ea580c"

LAYOUT_BASE = dict(
    template="plotly_white",
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family=FONT, color=TEXT, size=11),
    margin=dict(l=60, r=20, t=40, b=50),
    xaxis=dict(showgrid=True, gridcolor="#f1f5f9",
               linecolor=BORDER, tickfont=dict(family=FONT)),
    yaxis=dict(showgrid=True, gridcolor="#f1f5f9",
               linecolor=BORDER, tickfont=dict(family=FONT)),
    legend=dict(bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1,
                font=dict(family=FONT, size=10)),
    title=dict(font=dict(family=FONT, size=12, color=MUTED)),
)

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

# ── Load data ─────────────────────────────────────────────────────────────────
SIM    = pd.read_csv("outputs/processed_data/batch7_layered_simulation.csv")
SPARSE = None
PRED   = None

try:
    from loader import load_sparse_samples
    SPARSE = load_sparse_samples("data/result-batch_7.xlsx")
except Exception:
    pass

try:
    from ml_model import AdaptiveController, engineer_features
    from loader import load_sparse_samples
    b5   = load_sparse_samples("data/result-batch_5.xlsx")
    b7   = load_sparse_samples("data/result-batch_7.xlsx")
    ctrl = AdaptiveController()
    ctrl.train([b5, b7])
    PRED = ctrl.predict_yield(b7)
except Exception as e:
    print(f"ML load warning: {e}")

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.title = "AI Controller Dashboard"

app.layout = html.Div([
    html.Div([
        html.H1("AI Controller Dashboard",
                style={"margin": 0, "fontSize": "1.1rem", "fontFamily": FONT,
                       "fontWeight": "600", "color": TEXT}),
        html.P("Phase classification  ·  Yield prediction  ·  Control decisions",
               style={"margin": "2px 0 0", "fontSize": "0.8rem",
                      "fontFamily": FONT, "color": MUTED}),
    ], style={"padding": "14px 28px", "borderBottom": f"1px solid {BORDER}",
              "backgroundColor": SURFACE,
              "boxShadow": "0 1px 3px rgba(0,0,0,0.06)"}),

    dcc.Tabs(id="tabs", value="yield", children=[
        dcc.Tab(label="Yield Prediction",     value="yield",
                style=TAB_S, selected_style=TAB_SEL),
        dcc.Tab(label="Phase Classification", value="phase",
                style=TAB_S, selected_style=TAB_SEL),
        dcc.Tab(label="Control Decisions",    value="control",
                style=TAB_S, selected_style=TAB_SEL),
        dcc.Tab(label="Calibration",          value="calib",
                style=TAB_S, selected_style=TAB_SEL),
    ], style={"fontFamily": FONT, "backgroundColor": BG,
              "borderBottom": f"1px solid {BORDER}"}),

    html.Div(id="tab-content",
             style={"padding": "20px 28px", "backgroundColor": BG,
                    "minHeight": "80vh"}),

], style={"fontFamily": FONT, "backgroundColor": BG})


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):

    # ── Yield prediction ──────────────────────────────────────────────────────
    if tab == "yield":
        fig = go.Figure()

        if PRED is not None:
            fig.add_trace(go.Scatter(
                x=pd.concat([PRED.time_h, PRED.time_h[::-1]]),
                y=pd.concat([PRED.L1_upper, PRED.L1_lower[::-1]]),
                fill="toself", fillcolor="rgba(37,99,235,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% confidence",
            ))
            fig.add_trace(go.Scatter(
                x=PRED.time_h, y=PRED.L1_pred,
                line=dict(color=BLUE, width=2),
                name="Predicted L1 yield",
            ))

        if SPARSE is not None and "L1_yield_mgL" in SPARSE.columns:
            actual = SPARSE.dropna(subset=["L1_yield_mgL"])
            fig.add_trace(go.Scatter(
                x=actual.time_h, y=actual.L1_yield_mgL,
                mode="markers",
                marker=dict(color=TEXT, size=7, symbol="circle"),
                name="Actual L1 yield",
            ))

        fig.update_layout(**LAYOUT_BASE,
                          xaxis_title="Time (h)",
                          yaxis_title="L1 Yield (mg/L)",
                          title=dict(text="L1 Yield Prediction vs Actual"))

        mae = err_pct = final_pred = final_actual = 0
        if PRED is not None and SPARSE is not None:
            actual = SPARSE.dropna(subset=["L1_yield_mgL"])
            merged = actual.merge(PRED, on="time_h", how="inner")
            if not merged.empty:
                mae         = abs(merged.L1_yield_mgL - merged.L1_pred).mean()
                final_pred  = PRED.L1_pred.iloc[-1]
                final_actual = SPARSE.L1_yield_mgL.dropna().iloc[-1]
                err_pct     = abs(final_pred - final_actual) / final_actual * 100

        return html.Div([
            html.Div([
                _metric("Final Predicted", f"{final_pred:.0f} mg/L"),
                _metric("Final Actual",    f"{final_actual:.0f} mg/L"),
                _metric("Error",           f"{err_pct:.1f}%"),
                _metric("MAE",             f"{mae:.0f} mg/L"),
                _metric("Training points", "18"),
            ], style={"display": "flex", "gap": "12px", "marginBottom": "16px",
                      "flexWrap": "wrap"}),
            dcc.Graph(figure=fig, style={"height": "58vh"}),
        ])

    # ── Phase classification ──────────────────────────────────────────────────
    elif tab == "phase":
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=["Predicted Phase + Confidence",
                            "Actual vs Predicted"],
            vertical_spacing=0.12,
        )

        if not SIM.empty:
            ml_rows = SIM[SIM.ml_confidence > 0]
            fig.add_trace(go.Scatter(
                x=ml_rows.time_h, y=ml_rows.ml_confidence,
                line=dict(color=BLUE, width=1.5),
                name="ML confidence", fill="tozeroy",
                fillcolor="rgba(37,99,235,0.08)",
            ), row=1, col=1)

            phase_colors = {
                "growth":     "rgba(22,163,74,0.12)",
                "induction":  "rgba(234,179,8,0.12)",
                "production": "rgba(37,99,235,0.12)",
                "harvest":    "rgba(220,38,38,0.12)",
            }
            for phase, color in phase_colors.items():
                mask = SIM[SIM.phase == phase]
                if not mask.empty:
                    fig.add_vrect(
                        x0=mask.time_h.min(), x1=mask.time_h.max(),
                        fillcolor=color, line_width=0,
                        annotation_text=phase,
                        annotation_position="top left",
                        row=2, col=1,
                    )

            for layer, color in [("rules", TEXT), ("ml", BLUE), ("safety", RED)]:
                rows = SIM[SIM.layer_used == layer]
                fig.add_trace(go.Scatter(
                    x=rows.time_h,
                    y=[layer] * len(rows),
                    mode="markers",
                    marker=dict(color=color, size=8, symbol="square"),
                    name=f"Layer: {layer}",
                ), row=2, col=1)

        fig.update_layout(
            **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
            height=500,
        )
        fig.update_xaxes(title_text="Time (h)", row=2, col=1)
        fig.update_yaxes(title_text="Confidence", row=1, col=1)

        return dcc.Graph(figure=fig, style={"height": "65vh"})

    # ── Control decisions ─────────────────────────────────────────────────────
    elif tab == "control":
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=["L1 Yield Prediction (mg/L)",
                            "Methanol Pulse Events"],
            vertical_spacing=0.14,
            row_heights=[0.65, 0.35],
        )

        if not SIM.empty:
            pred_rows = SIM.dropna(subset=["L1_predicted"])
            if not pred_rows.empty:
                fig.add_trace(go.Scatter(
                    x=pred_rows.time_h, y=pred_rows.L1_predicted,
                    line=dict(color=BLUE, width=2),
                    name="L1 predicted",
                ), row=1, col=1)

            if "L1_yield_mgL_actual" in SIM.columns:
                actual = SIM.dropna(subset=["L1_yield_mgL_actual"])
                fig.add_trace(go.Scatter(
                    x=actual.time_h, y=actual.L1_yield_mgL_actual,
                    mode="markers+lines",
                    line=dict(color=TEXT, width=1.5, dash="dot"),
                    marker=dict(color=TEXT, size=8, symbol="circle"),
                    name="L1 actual",
                ), row=1, col=1)

            pulse_times = SIM.loc[SIM.methanol_feed_pct > 0, "time_h"]
            if not pulse_times.empty:
                for t in pulse_times:
                    fig.add_shape(
                        type="line", x0=t, x1=t, y0=0, y1=1,
                        line=dict(color=ORANGE, width=2),
                        row=2, col=1,
                    )
                fig.add_trace(go.Scatter(
                    x=pulse_times, y=[1.0] * len(pulse_times),
                    mode="markers",
                    marker=dict(color=ORANGE, size=10, symbol="circle"),
                    name="MeOH pulse",
                ), row=2, col=1)

        fig.update_layout(
            **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
            height=550,
        )
        fig.update_xaxes(title_text="Time (h)", row=2, col=1)
        fig.update_yaxes(title_text="mg/L", row=1, col=1)
        fig.update_yaxes(title_text="Pulse on/off", tickvals=[0, 1],
                         ticktext=["off", "on"], range=[-0.1, 1.3],
                         row=2, col=1)

        actions = SIM[SIM.action != "none"][
            ["time_h", "phase", "layer_used", "action", "message"]
        ].head(20)

        return html.Div([
            dcc.Graph(figure=fig, style={"height": "55vh"}),
            html.H4("Key Actions",
                    style={"fontFamily": FONT, "fontWeight": "500",
                           "color": MUTED, "marginTop": "20px",
                           "fontSize": "0.9rem", "textTransform": "uppercase",
                           "letterSpacing": "0.06em"}),
            _table(actions),
        ])

    # ── Calibration ───────────────────────────────────────────────────────────
    elif tab == "calib":
        mu_max = g0 = x0 = 0.0

        if SPARSE is not None:
            dcw = SPARSE.dropna(subset=["DCW_gL"])
            if len(dcw) >= 2:
                dt     = dcw.time_h.diff()
                mu     = (np.log(dcw.DCW_gL / dcw.DCW_gL.shift(1)) / dt
                          ).replace([np.inf, -np.inf], np.nan).dropna()
                mu_max = min(float(mu.quantile(0.9)), 0.25)
            if "glycerol_gL" in SPARSE.columns:
                g0 = float(SPARSE.glycerol_gL.dropna().iloc[0])
            if "DCW_gL" in SPARSE.columns:
                x0 = float(SPARSE.DCW_gL.dropna().iloc[0])

        return html.Div([
            html.Div([
                _metric("mu_max (capped)",      f"{mu_max:.4f} h⁻¹"),
                _metric("Induction threshold",  f"{0.15 * mu_max:.4f} h⁻¹"),
                _metric("Glycerol₀",       f"{g0:.2f} g/L"),
                _metric("Depletion threshold",  f"{0.04 * g0:.2f} g/L"),
                _metric("X₀",              f"{x0:.2f} g/L"),
                _metric("Calibration window",   "12 h"),
                _metric("ML confidence min",    "0.85"),
                _metric("Training batches",     "2"),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "12px"}),

            html.Hr(style={"borderColor": BORDER, "margin": "20px 0"}),
            html.P(
                "Calibration uses the first 12 h of each new run to estimate "
                "mu_max, initial glycerol, and biomass. These values set "
                "scale-agnostic thresholds for the rule-based layer, making "
                "the controller work across different bioreactor volumes.",
                style={"fontFamily": FONT, "color": MUTED,
                       "maxWidth": "600px", "lineHeight": "1.6",
                       "fontSize": "0.875rem"},
            ),
        ])


def _metric(label, value):
    return html.Div([
        html.Div(label, style={
            "fontFamily": FONT, "fontSize": "0.7rem", "color": MUTED,
            "textTransform": "uppercase", "letterSpacing": "0.05em",
            "fontWeight": "600",
        }),
        html.Div(value, style={
            "fontFamily": FONT, "fontSize": "1.3rem",
            "color": TEXT, "fontWeight": "500", "marginTop": "2px",
        }),
    ], style={
        "padding": "12px 16px", "backgroundColor": SURFACE,
        "border": f"1px solid {BORDER}", "borderRadius": "6px",
        "minWidth": "130px",
        "boxShadow": "0 1px 2px rgba(0,0,0,0.04)",
    })


def _table(df):
    return html.Table([
        html.Thead(html.Tr([
            html.Th(c, style={
                "fontFamily": FONT, "fontSize": "0.7rem",
                "textTransform": "uppercase", "color": MUTED,
                "padding": "8px 12px", "textAlign": "left",
                "borderBottom": f"2px solid {BORDER}",
                "fontWeight": "600", "letterSpacing": "0.05em",
            }) for c in df.columns
        ])),
        html.Tbody([
            html.Tr([
                html.Td(str(v)[:60], style={
                    "fontFamily": FONT, "fontSize": "0.83rem", "color": TEXT,
                    "padding": "6px 12px", "borderBottom": f"1px solid {BORDER}",
                }) for v in row
            ]) for _, row in df.iterrows()
        ])
    ], style={"width": "100%", "borderCollapse": "collapse",
              "backgroundColor": SURFACE,
              "borderRadius": "6px", "overflow": "hidden"})


if __name__ == "__main__":
    print("AI Dashboard at http://127.0.0.1:8051")
    app.run(debug=True, port=8051)
