import pandas as pd
import numpy as np
from pathlib import Path
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
sys.path.insert(0, "src") 

FONT = "Times New Roman"
BG   = "#ffffff"
BG2  = "#fafafa"
BORDER = "#dddddd"
BLUE   = "#1a56db"
BLACK  = "#111111"
GRAY   = "#666666"
GREEN  = "#16a34a"
RED    = "#dc2626"
ORANGE = "#ea580c"

def base_layout(title=""):
    return dict(
        template="plotly_white",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family=FONT, color=BLACK, size=11),
        margin=dict(l=60, r=20, t=40, b=50),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0",
                   linecolor=BLACK, tickfont=dict(family=FONT)),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0",
                   linecolor=BLACK, tickfont=dict(family=FONT)),
        legend=dict(bgcolor=BG, bordercolor=BORDER, borderwidth=1,
                    font=dict(family=FONT, size=10)),
        title=dict(text=title, font=dict(family=FONT, size=12, color=GRAY)),
    )

# ── Load data ─────────────────────────────────────────────────────────────────
SIM   = pd.read_csv("outputs/processed_data/batch7_layered_simulation.csv")
SPARSE = None
PRED   = None

try:
    from loader import load_sparse_samples
    SPARSE = load_sparse_samples("data/result-batch_7.xlsx")
except:
    pass

try:
    from ml_model import AdaptiveController, engineer_features
    from loader import load_sparse_samples
    b5 = load_sparse_samples("data/result-batch_5.xlsx")
    b7 = load_sparse_samples("data/result-batch_7.xlsx")
    ctrl = AdaptiveController()
    ctrl.train([b5, b7])
    PRED = ctrl.predict_yield(b7)
except Exception as e:
    print(f"ML load warning: {e}")

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.title = "AI Controller Dashboard"

LABEL = dict(fontFamily=FONT, fontSize="0.7rem", textTransform="uppercase",
             letterSpacing="0.08em", color=GRAY, display="block",
             marginBottom="4px", marginTop="12px")

app.layout = html.Div([

    # Header
    html.Div([
        html.H1("AI Controller Dashboard",
                style=dict(margin=0, fontSize="1.3rem",
                           fontFamily=FONT, fontWeight="normal", color=BLACK)),
        html.P("Phase classification · Yield prediction · Control decisions",
               style=dict(margin="2px 0 0", fontSize="0.82rem",
                          fontFamily=FONT, color=GRAY)),
    ], style=dict(padding="16px 28px", borderBottom=f"2px solid {BLACK}",
                  backgroundColor=BG)),

    # Tabs
    dcc.Tabs(id="tabs", value="yield", children=[
        dcc.Tab(label="Yield Prediction",     value="yield"),
        dcc.Tab(label="Phase Classification", value="phase"),
        dcc.Tab(label="Control Decisions",    value="control"),
        dcc.Tab(label="Calibration",          value="calib"),
    ], style=dict(fontFamily=FONT, fontSize="0.9rem")),

    html.Div(id="tab-content",
             style=dict(padding="20px", backgroundColor=BG,
                        minHeight="80vh")),

], style=dict(fontFamily=FONT, backgroundColor=BG))


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):

    # ── Yield prediction ──────────────────────────────────────────────────────
    if tab == "yield":
        fig = go.Figure()

        # Uncertainty band
        if PRED is not None:
            fig.add_trace(go.Scatter(
                x=pd.concat([PRED.time_h, PRED.time_h[::-1]]),
                y=pd.concat([PRED.L1_upper, PRED.L1_lower[::-1]]),
                fill="toself", fillcolor="rgba(26,86,219,0.1)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% confidence", showlegend=True,
            ))
            fig.add_trace(go.Scatter(
                x=PRED.time_h, y=PRED.L1_pred,
                line=dict(color=BLUE, width=2),
                name="Predicted L1 yield",
            ))

        # Actual measurements
        if SPARSE is not None and "L1_yield_mgL" in SPARSE.columns:
            actual = SPARSE.dropna(subset=["L1_yield_mgL"])
            fig.add_trace(go.Scatter(
                x=actual.time_h, y=actual.L1_yield_mgL,
                mode="markers", marker=dict(color=BLACK, size=7, symbol="circle"),
                name="Actual L1 yield",
            ))

        fig.update_layout(**base_layout("L1 Yield Prediction vs Actual"),
                          xaxis_title="Time (h)",
                          yaxis_title="L1 Yield (mg/L)")

        # Metrics
        if PRED is not None and SPARSE is not None:
            actual = SPARSE.dropna(subset=["L1_yield_mgL"])
            merged = actual.merge(PRED, on="time_h", how="inner")
            if not merged.empty:
                mae = abs(merged.L1_yield_mgL - merged.L1_pred).mean()
                final_pred   = PRED.L1_pred.iloc[-1]
                final_actual = SPARSE.L1_yield_mgL.dropna().iloc[-1]
                err_pct = abs(final_pred - final_actual) / final_actual * 100
            else:
                mae, err_pct, final_pred, final_actual = 0, 0, 0, 0
        else:
            mae = err_pct = final_pred = final_actual = 0

        return html.Div([
            html.Div([
                _metric("Final Predicted", f"{final_pred:.0f} mg/L"),
                _metric("Final Actual",    f"{final_actual:.0f} mg/L"),
                _metric("Error",           f"{err_pct:.1f}%"),
                _metric("MAE",             f"{mae:.0f} mg/L"),
                _metric("Training points", "18"),
            ], style=dict(display="flex", gap="16px", marginBottom="16px")),
            dcc.Graph(figure=fig, style=dict(height="60vh")),
        ])

    # ── Phase classification ──────────────────────────────────────────────────
    elif tab == "phase":
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=["Predicted Phase + Confidence",
                                            "Actual vs Predicted"],
                            vertical_spacing=0.12)

        if not SIM.empty:
            # Confidence line
            ml_rows = SIM[SIM.ml_confidence > 0]
            fig.add_trace(go.Scatter(
                x=ml_rows.time_h, y=ml_rows.ml_confidence,
                line=dict(color=BLUE, width=1.5),
                name="ML confidence", fill="tozeroy",
                fillcolor="rgba(26,86,219,0.1)",
            ), row=1, col=1)

            # Phase background colors
            phase_colors = {"growth": "rgba(34,197,94,0.15)",
                            "induction": "rgba(234,179,8,0.15)",
                            "production": "rgba(59,130,246,0.15)",
                            "harvest": "rgba(239,68,68,0.15)"}

            for phase, color in phase_colors.items():
                mask = SIM[SIM.phase == phase]
                if not mask.empty:
                    fig.add_vrect(
                        x0=mask.time_h.min(), x1=mask.time_h.max(),
                        fillcolor=color, line_width=0,
                        annotation_text=phase, annotation_position="top left",
                        row=2, col=1,
                    )

            # Layer used
            for layer, color in [("rules", BLACK), ("ml", BLUE),
                                  ("safety", RED)]:
                rows = SIM[SIM.layer_used == layer]
                fig.add_trace(go.Scatter(
                    x=rows.time_h,
                    y=[layer] * len(rows),
                    mode="markers",
                    marker=dict(color=color, size=8, symbol="square"),
                    name=f"Layer: {layer}",
                ), row=2, col=1)

        fig.update_layout(**{k: v for k, v in
                             base_layout("Phase Classification").items()
                             if k not in ("xaxis", "yaxis")},
                          height=500)
        fig.update_xaxes(title_text="Time (h)", row=2, col=1)
        fig.update_yaxes(title_text="Confidence", row=1, col=1)

        return dcc.Graph(figure=fig, style=dict(height="65vh"))

    # ── Control decisions ─────────────────────────────────────────────────────
    elif tab == "control":
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=["L1 Yield Prediction (mg/L)",
                                            "Methanol Pulse Events"],
                            vertical_spacing=0.14,
                            row_heights=[0.65, 0.35])

        if not SIM.empty:
            # L1 predicted trajectory
            pred_rows = SIM.dropna(subset=["L1_predicted"])
            if not pred_rows.empty:
                fig.add_trace(go.Scatter(
                    x=pred_rows.time_h, y=pred_rows.L1_predicted,
                    line=dict(color=BLUE, width=2),
                    name="L1 predicted",
                ), row=1, col=1)

            # Actual L1 measurements as bold markers
            if "L1_yield_mgL_actual" in SIM.columns:
                actual = SIM.dropna(subset=["L1_yield_mgL_actual"])
                fig.add_trace(go.Scatter(
                    x=actual.time_h, y=actual.L1_yield_mgL_actual,
                    mode="markers+lines",
                    line=dict(color=BLACK, width=1.5, dash="dot"),
                    marker=dict(color=BLACK, size=8, symbol="circle"),
                    name="L1 actual",
                ), row=1, col=1)

            # Methanol pulses as stem markers (visible even for sparse binary data)
            pulse_times = SIM.loc[SIM.methanol_feed_pct > 0, "time_h"]
            if not pulse_times.empty:
                # Stems: vertical lines from 0 → 1
                for t in pulse_times:
                    fig.add_shape(
                        type="line",
                        x0=t, x1=t, y0=0, y1=1,
                        line=dict(color=ORANGE, width=2),
                        row=2, col=1,
                    )
                # Dots at top of each stem
                fig.add_trace(go.Scatter(
                    x=pulse_times, y=[1.0] * len(pulse_times),
                    mode="markers",
                    marker=dict(color=ORANGE, size=10, symbol="circle"),
                    name="MeOH pulse",
                ), row=2, col=1)

        fig.update_layout(**{k: v for k, v in
                             base_layout("Control Decisions").items()
                             if k not in ("xaxis", "yaxis")},
                          height=550)
        fig.update_xaxes(title_text="Time (h)", row=2, col=1)
        fig.update_yaxes(title_text="mg/L", row=1, col=1)
        fig.update_yaxes(title_text="Pulse on/off", tickvals=[0, 1],
                         ticktext=["off", "on"], range=[-0.1, 1.3],
                         row=2, col=1)

        # Action summary table
        actions = SIM[SIM.action != "none"][
            ["time_h", "phase", "layer_used", "action", "message"]
        ].head(20)

        return html.Div([
            dcc.Graph(figure=fig, style=dict(height="55vh")),
            html.H4("Key Actions", style=dict(fontFamily=FONT,
                                               fontWeight="normal",
                                               color=GRAY, marginTop="16px")),
            _table(actions),
        ])

    # ── Calibration ───────────────────────────────────────────────────────────
    elif tab == "calib":
        rows = []
        if SPARSE is not None:
            dcw = SPARSE.dropna(subset=["DCW_gL"])
            if len(dcw) >= 2:
                dt   = dcw.time_h.diff()
                mu   = (np.log(dcw.DCW_gL / dcw.DCW_gL.shift(1)) / dt
                        ).replace([np.inf, -np.inf], np.nan).dropna()
                mu_max = min(float(mu.quantile(0.9)), 0.25)
            else:
                mu_max = 0.0
            g0 = SPARSE.glycerol_gL.dropna().iloc[0] \
                 if "glycerol_gL" in SPARSE.columns else 0
            x0 = SPARSE.DCW_gL.dropna().iloc[0] \
                 if "DCW_gL" in SPARSE.columns else 0

        return html.Div([
            html.Div([
                _metric("μ_max (capped)",    f"{mu_max:.4f} h⁻¹"),
                _metric("Induction threshold", f"{0.15*mu_max:.4f} h⁻¹"),
                _metric("Glycerol₀",         f"{g0:.2f} g/L"),
                _metric("Depletion threshold", f"{0.04*g0:.2f} g/L"),
                _metric("X₀",               f"{x0:.2f} g/L"),
                _metric("Calibration window", "12 h"),
                _metric("ML confidence min",  "0.85"),
                _metric("Training batches",   "2"),
            ], style=dict(display="flex", flexWrap="wrap", gap="16px")),

            html.Hr(style=dict(borderColor=BORDER, margin="20px 0")),
            html.P("Calibration uses the first 12h of each new run to estimate "
                   "μ_max, initial glycerol, and biomass. These values set "
                   "scale-agnostic thresholds for the rule-based layer, making "
                   "the controller work across different bioreactor volumes.",
                   style=dict(fontFamily=FONT, color=GRAY, maxWidth="600px")),
        ])


def _metric(label, value):
    return html.Div([
        html.Div(label, style=dict(fontFamily=FONT, fontSize="0.7rem",
                                   color=GRAY, textTransform="uppercase",
                                   letterSpacing="0.05em")),
        html.Div(value, style=dict(fontFamily=FONT, fontSize="1.4rem",
                                   color=BLACK, fontWeight="500")),
    ], style=dict(padding="12px 16px", border=f"1px solid {BORDER}",
                  borderRadius="6px", minWidth="130px"))


def _table(df):
    return html.Table([
        html.Thead(html.Tr([
            html.Th(c, style=dict(fontFamily=FONT, fontSize="0.75rem",
                                  textTransform="uppercase", color=GRAY,
                                  padding="8px", textAlign="left",
                                  borderBottom=f"1px solid {BORDER}"))
            for c in df.columns
        ])),
        html.Tbody([
            html.Tr([
                html.Td(str(v)[:60], style=dict(fontFamily=FONT,
                                                 fontSize="0.82rem",
                                                 padding="6px 8px",
                                                 borderBottom=f"1px solid {BORDER}"))
                for v in row
            ]) for _, row in df.iterrows()
        ])
    ], style=dict(width="100%", borderCollapse="collapse"))


if __name__ == "__main__":
    print("AI Dashboard at http://127.0.0.1:8051")
    app.run(debug=True, port=8051)
