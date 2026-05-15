# Bioreactor Control System

A Python-based system for bioreactor monitoring, control, and RS-232 sensor auto-detection.

---

## Project Structure

```
bioreactor/
├── api/                    # FastAPI backend
│   └── main.py
├── src/
│   ├── control.py          # Control logic and dataclasses (SensorReading, ControlOutput)
│   ├── dashboard.py        # Main process dashboard
│   ├── ai_dashboard.py     # AI-assisted dashboard
│   ├── features.py         # Feature engineering
│   ├── loader.py           # Data loading utilities
│   ├── ml_model.py         # ML model
│   ├── preprocess.py       # Data preprocessing
│   ├── transforms.py       # Signal transforms
│   ├── serial_detector.py  # RS-232 sensor detection engine
│   └── sensor_dashboard.py # Live sensor detection dashboard (Dash)
├── data/
│   ├── sensor_signatures.json   # Learned sensor signatures (auto-created)
│   └── *.xlsx                   # Batch data files
├── detect_sensors.py       # CLI tool for sensor detection
├── requirements.txt
├── docker-compose.yml
├── Dockerfile.api
└── Dockerfile.web
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## RS-232 Sensor Detection

Automatically detects what sensor is connected to each RS-232 port when you plug it in.

### How it works

1. **Scan** — lists all serial ports on the computer
2. **Probe** — for each port, tries every combination of baud rate and serial config (see tables below)
3. **Passive listen first** — opens the port, waits for the sensor to stream data on its own (many old sensors do this); only sends query commands if nothing arrives
4. **Readability check** — automatically skips garbled data caused by wrong baud rate or parity, no manual intervention needed
5. **Match** — compares the readable response against built-in patterns and any user-saved signatures
6. **Label** — if a sensor is unrecognised, you type its name in the dashboard or CLI; the signature is saved and used automatically next time

**Baud rates tried** (ordered by likelihood):

| Rate | Note |
|------|------|
| 9600 | Most common default |
| 4800 | Common on older instruments |
| 19200 | Mid-range |
| 2400 | Legacy |
| 38400 | Mid-range |
| 1200 | Old slow sensors |
| 57600 | Fast |
| 115200 | Fast |
| 600 | Very old hardware |
| 300 | Very old hardware |

**Serial configurations tried** (for each baud rate):

| Config | Bits | Parity | Stop bits | Common on |
|--------|------|--------|-----------|-----------|
| 8N1 | 8 | None | 1 | Most modern sensors |
| 7E1 | 7 | Even | 1 | Old industrial instruments |
| 7O1 | 7 | Odd | 1 | Old industrial instruments |
| 8E1 | 8 | Even | 1 | Some lab equipment |

### Supported sensor types (built-in patterns)

| Sensor      | Example response match          | Color  |
|-------------|----------------------------------|--------|
| pH          | `pH=7.02`, `pH: 6.8`            | Green  |
| DO          | `DO=98.5`, `O2=21.0`, `air sat` | Blue   |
| Temperature | `T=37.0 C`, `Temp: 36.8`        | Orange |
| Foam        | `foam`, `level`, `F=1`          | Purple |
| Stirrer     | `RPM=700`, `stir`               | Cyan   |
| Flow        | `flow=5.2`, `ml/min`            | Red    |

Any sensor not matching the above is shown as **UNKNOWN** and can be labelled manually.

---

### Option 1 — Live Dashboard (recommended)

```bash
source venv/bin/activate
python src/sensor_dashboard.py
```

Open **http://localhost:8060** in your browser.

The dashboard:
- Shows one card per connected serial port, refreshing every 3 seconds
- Detects plug and unplug events automatically
- Displays sensor type, baud rate, connection status, and raw response preview
- Shows a label input box for UNKNOWN sensors — type the sensor name and click **Save label**

**Status badge colours:**

| Status        | Meaning                                       |
|---------------|-----------------------------------------------|
| CONNECTED     | Port responded and sensor type was identified |
| PROBING       | Currently sending test commands               |
| NO RESPONSE   | Port found but device sent nothing back       |
| DISCONNECTED  | Device was unplugged                          |

---

### Option 2 — CLI tool

```bash
source venv/bin/activate
python detect_sensors.py
```

**Commands:**

```bash
# Scan all ports once and print results
python detect_sensors.py

# Watch for plug/unplug events continuously (Ctrl-C to stop)
python detect_sensors.py --monitor

# Interactively label an unknown sensor
python detect_sensors.py --label

# List all saved sensor signatures
python detect_sensors.py --list
```

---

### Sensor signature learning

When you label a sensor (via dashboard or `--label`), the signature is saved to:

```
data/sensor_signatures.json
```

Example saved entry:

```json
{
  "device": "/dev/ttyUSB0",
  "description": "USB-Serial Controller",
  "vid": 1027,
  "pid": 24577,
  "baud": 9600,
  "raw_sample": "pH=7.02\r\n",
  "sensor_type": "pH",
  "pattern": "pH"
}
```

Next time the same device is plugged in, it is recognised immediately without probing.

To add a more specific regex pattern (instead of the default `.*`), use the `--label` CLI command which prompts for a pattern.

---

### Key files

| File | Purpose |
|------|---------|
| `src/serial_detector.py` | Core engine — probe ports, match signatures, hot-plug monitor |
| `src/sensor_dashboard.py` | Dash web dashboard, auto-refreshes every 3 s |
| `detect_sensors.py` | CLI tool for one-shot scan, monitor mode, and labelling |
| `data/sensor_signatures.json` | Persisted sensor signatures (created automatically) |

---

## Control System

The bioreactor control system is in `src/control.py`. It uses a layered control approach:

- **Safety layer** — hard limits on pH, DO, temperature
- **Rules layer** — phase-based logic (batch, fed-batch, induction)
- **ML layer** — adaptive predictions for feed rate and L1 yield

Sensor readings use the `SensorReading` dataclass; control decisions use `ControlOutput`.

---

## Docker

```bash
docker-compose up --build
```

Runs the API (`Dockerfile.api`) and web frontend (`Dockerfile.web`) together.
