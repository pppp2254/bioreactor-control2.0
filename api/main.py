import sys
sys.path.insert(0, "src")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import pandas as pd
import numpy as np
import json

from loader import load_batch, load_sparse_samples
from features import compute_growth_rate, detect_phases
from control import BioreactorController, SensorReading, simulate_batch
from ml_model import AdaptiveController

app = FastAPI(title="Bioreactor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auto-load all batches from data/ ─────────────────────────────────────────
BATCHES = {}   # batch_name → continuous DataFrame
SPARSE  = {}   # batch_name → sparse DataFrame

@app.on_event("startup")
def load_data():
    data_dir = Path("data")
    for f in sorted(data_dir.glob("*.xlsx")):
        key = f.stem
        try:
            sheets = load_batch(f)
            df = list(sheets.values())[0]
            run = df[df.phase == "run"].copy()
            # Auto-detect end
            diffs = run.time_h.diff().dropna()
            large = diffs[diffs > 1.0]
            if not large.empty:
                end_h = float(run.loc[large.index[0] - 1, "time_h"])
                run = run[run.time_h <= end_h]
            BATCHES[key] = run
            sp = load_sparse_samples(f)
            if sp is not None:
                SPARSE[key] = sp
            print(f"Loaded {key}: {len(run)} continuous rows, "
                  f"{len(sp) if sp is not None else 0} sparse rows")
        except Exception as e:
            print(f"Failed {key}: {e}")


def df_to_json(df: pd.DataFrame) -> list:
    df = df.loc[:, ~df.columns.duplicated()]
    return json.loads(
        df.replace([np.inf, -np.inf], np.nan)
          .fillna("null")
          .to_json(orient="records")
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "batches": list(BATCHES.keys())}


@app.get("/batches")
def get_batches():
    return {
        k: {
            "rows":       len(v),
            "columns":    list(v.columns),
            "time_range": [float(v.time_h.min()), float(v.time_h.max())],
            "has_sparse": k in SPARSE,
            "sparse_rows": len(SPARSE[k]) if k in SPARSE else 0,
        }
        for k, v in BATCHES.items()
    }


@app.get("/signals/{batch}")
def get_signals(batch: str, cols: str = "all"):
    if batch not in BATCHES:
        return {"error": f"batch '{batch}' not found — available: {list(BATCHES.keys())}"}
    df = BATCHES[batch]
    if cols != "all":
        wanted = ["time_h"] + [c for c in cols.split(",") if c in df.columns]
        df = df[wanted]
    return df_to_json(df)


@app.get("/sparse/{batch}")
def get_sparse(batch: str):
    if batch not in SPARSE:
        return {"error": f"no sparse data for '{batch}'"}
    return df_to_json(SPARSE[batch])


@app.get("/features/{batch}")
def get_features(batch: str):
    if batch not in SPARSE:
        return {"error": "no sparse data"}
    sp     = SPARSE[batch]
    events = detect_phases(sp)
    gr     = compute_growth_rate(sp)
    return {"events": events, "growth_rate": df_to_json(gr)}

@app.get("/transforms/{batch}")
def get_transforms(batch: str):
    if batch not in BATCHES:
        return {"error": f"batch not found"}
    from transforms import run_adaptive_transforms
    df = BATCHES[batch]
    results = run_adaptive_transforms(df, batch_name=batch)
    # Convert numpy arrays to lists for JSON serialization
    import json
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj
    return make_serializable(results)


@app.get("/simulate/{batch}")
def get_simulation(batch: str):
    if batch not in SPARSE:
        return {"error": "no sparse data"}
    sparse = SPARSE[batch]
    cont   = BATCHES.get(batch)
    sim    = simulate_batch(sparse, cont, load_ml=True)
    return df_to_json(sim)


@app.get("/predict/{batch}")
def get_prediction(batch: str):
    if batch not in SPARSE:
        return {"error": "no sparse data"}
    try:
        sparse_list = [v for v in SPARSE.values()]
        ctrl = AdaptiveController()
        ctrl.train(sparse_list)
        pred = ctrl.predict_yield(SPARSE[batch])
        return df_to_json(pred)
    except Exception as e:
        return {"error": str(e)}


@app.get("/available")
def get_available():
    """Return what signals are available per batch — for UI to auto-configure."""
    result = {}
    for key, df in BATCHES.items():
        continuous = [c for c in df.columns
                      if df[c].notna().sum() > len(df) * 0.5
                      and c not in ("datetime", "phase")]
        sparse_cols = list(SPARSE[key].columns) if key in SPARSE else []
        result[key] = {
            "continuous": continuous,
            "sparse":     sparse_cols,
            "time_range": [float(df.time_h.min()), float(df.time_h.max())],
        }
    return result


# ── Control step ──────────────────────────────────────────────────────────────

class StepRequest(BaseModel):
    batch:       str
    time_h:      float
    DO_pct:      Optional[float] = None
    pH:          Optional[float] = None
    temp_c:      Optional[float] = None
    stirrer:     Optional[float] = None
    glycerol_gL: Optional[float] = None
    methanol_gL: Optional[float] = None
    DCW_gL:      Optional[float] = None
    OD660:       Optional[float] = None

_controllers = {}

@app.post("/step")
def control_step(req: StepRequest):
    if req.batch not in _controllers:
        ctrl = BioreactorController()
        ctrl.load_models()
        if req.batch in SPARSE:
            ctrl.calibrate(SPARSE[req.batch])
        _controllers[req.batch] = ctrl

    ctrl    = _controllers[req.batch]
    reading = SensorReading(
        time_h      = req.time_h,
        DO_pct      = req.DO_pct,
        pH          = req.pH,
        temp_c      = req.temp_c,
        stirrer     = req.stirrer,
        glycerol_gL = req.glycerol_gL,
        methanol_gL = req.methanol_gL,
        DCW_gL      = req.DCW_gL,
        OD660       = req.OD660,
    )
    out = ctrl.step(reading, sparse_history=SPARSE.get(req.batch))
    return {
        "time_h":            out.time_h,
        "phase":             out.phase,
        "layer_used":        out.layer_used,
        "action":            out.action,
        "methanol_feed_pct": out.methanol_feed_pct,
        "stirrer_setpoint":  out.stirrer_setpoint,
        "message":           out.message,
        "alerts":            out.alerts,
        "ml_phase":          out.ml_phase,
        "ml_confidence":     out.ml_confidence,
        "L1_predicted":      out.L1_predicted,
    }


@app.post("/reset/{batch}")
def reset_controller(batch: str):
    if batch in _controllers:
        del _controllers[batch]
    return {"status": "reset", "batch": batch}


@app.post("/upload")
async def upload_batch(file: bytes, filename: str):
    """Accept new batch file upload and auto-load it."""
    path = Path("data") / filename
    path.write_bytes(file)
    # Reload
    load_data()
    return {"status": "loaded", "batch": Path(filename).stem}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# ── File upload ───────────────────────────────────────────────────────────────
from fastapi import UploadFile, File

@app.post("/upload")
async def upload_batch(file: UploadFile = File(...)):
    """Upload a new batch Excel file and auto-load it."""
    if not file.filename.endswith(".xlsx"):
        return {"error": "Only .xlsx files accepted"}

    path = Path("data") / file.filename
    contents = await file.read()
    path.write_bytes(contents)

    # Reload this specific file
    key = path.stem
    try:
        sheets = load_batch(path)
        df = list(sheets.values())[0]
        run = df[df.phase == "run"].copy()
        diffs = run.time_h.diff().dropna()
        large = diffs[diffs > 1.0]
        if not large.empty:
            end_h = float(run.loc[large.index[0] - 1, "time_h"])
            run = run[run.time_h <= end_h]
        BATCHES[key] = run

        sp = load_sparse_samples(path)
        if sp is not None:
            SPARSE[key] = sp

        return {
            "status":       "loaded",
            "batch":        key,
            "continuous_rows": len(run),
            "sparse_rows":  len(sp) if sp is not None else 0,
            "time_range":   [float(run.time_h.min()), float(run.time_h.max())],
        }
    except Exception as e:
        return {"error": str(e)}
