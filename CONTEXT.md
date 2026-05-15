# Bioreactor Control System — Project Context

## Project Goal
Automated control system for HPV52 L1 protein production in H. polymorpha
Target: input desired yield → system calculates optimal feed control plan

## Data
- result-batch_5.xlsx: batch cultivation, 0-19h continuous, 0-66h sparse
- result-batch_7.xlsx: HCDC, 0-7h continuous, 0-120h sparse (full L1 yield data)
- Continuous signals: DO, pH, stirrer, temp, pumps (logged every 5min)
- Sparse signals: OD660, DCW, glycerol, methanol, L1 yield (manual samples every 6h)

## Architecture
- src/loader.py       — auto-detect and load any xlsx
- src/preprocess.py   — clean, spike removal, auto-detect experiment end
- src/features.py     — growth rate, phase detection, key events
- src/transforms.py   — adaptive FFT/CWT/Hilbert (window-aware, handles NaN)
- src/control.py      — layered controller: safety → rules → ML
- src/ml_model.py     — GP yield predictor + phase classifier
- src/dashboard.py    — Dash signal explorer (port 8050)
- src/ai_dashboard.py — Dash AI/ML dashboard (port 8051)
- api/main.py         — FastAPI REST API (port 8000)
- web/src/index.ts    — Elysia/Bun simulator UI (port 3000)

## Current Status
- Phase 1-4 complete
- STFT not yet added to transforms.py
- fillna(method=) deprecation bug in transforms.py line ~33
- Meeting with paper author next week — need more batch data

## Key Findings
- DO pulse interval: 1.75h (matches paper 1.4h protocol)
- Induction at 66h, glycerol depletion at 84h
- Peak L1: 2728 mg/L at 120h
- ML yield prediction: 7.2% error with 18 training points

## TODO
1. Add STFT to transforms.py
2. Fix fillna deprecation warnings
3. Fix docker-compose
4. Add biomass ceiling check to SafetyLayer
5. Prepare demo for author meeting
