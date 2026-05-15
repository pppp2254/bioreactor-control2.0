import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.signal import hilbert
import pywt
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_POINTS_FFT     = 30    # minimum points needed for meaningful FFT
MIN_POINTS_WAVELET = 20    # minimum points for wavelet
DT_H               = 1/12  # 5-minute sampling = 1/12 hour

# Methanol pulse frequency: paper feeds every ~1.4h → 0.714 cycles/h
METHANOL_PULSE_FREQ = 0.714  # cycles/h


# ── Adaptive window finder ────────────────────────────────────────────────────

def find_valid_windows(series: pd.Series, time_h: pd.Series,
                       min_window_h: float = 0.5) -> list:
    """
    Find contiguous non-NaN windows in a signal.
    Returns list of (start_idx, end_idx, duration_h) tuples.
    Only returns windows longer than min_window_h.
    """
    valid  = series.notna().values
    windows = []
    in_window = False
    start = 0

    for i, v in enumerate(valid):
        if v and not in_window:
            start = i
            in_window = True
        elif not v and in_window:
            end = i - 1
            duration = float(time_h.iloc[end] - time_h.iloc[start])
            if duration >= min_window_h:
                windows.append((start, end, duration))
            in_window = False

    # Handle window that extends to end
    if in_window:
        end = len(valid) - 1
        duration = float(time_h.iloc[end] - time_h.iloc[start])
        if duration >= min_window_h:
            windows.append((start, end, duration))

    return windows


def get_best_window(series: pd.Series, time_h: pd.Series,
                    min_window_h: float = 0.5):
    """Return the longest valid window of a signal."""
    windows = find_valid_windows(series, time_h, min_window_h)
    if not windows:
        return None, None, None
    # Pick longest window
    best = max(windows, key=lambda w: w[2])
    start, end, duration = best
    return (series.iloc[start:end+1].values,
            time_h.iloc[start:end+1].values,
            duration)


# ── Adaptive FFT ──────────────────────────────────────────────────────────────

def adaptive_fft(series: pd.Series, time_h: pd.Series,
                 label: str = "") -> dict:
    """
    FFT applied only to valid signal windows.
    Returns dict with frequencies, amplitudes, dominant periods,
    and metadata about the window used.
    """
    data, times, duration = get_best_window(series, time_h)

    result = {
        "signal":       label or series.name,
        "valid_window": None,
        "duration_h":   None,
        "n_points":     0,
        "freqs":        None,
        "amps":         None,
        "dominant":     [],
        "method":       None,
        "note":         "",
    }

    if data is None or len(data) < MIN_POINTS_FFT:
        n = len(data) if data is not None else 0
        result["note"] = (f"Window too short ({n} pts < {MIN_POINTS_FFT}) "
                          f"— using simple stats instead")
        result["method"] = "stats"
        if data is not None and len(data) > 2:
            result["stats"] = {
                "mean":  float(np.nanmean(data)),
                "std":   float(np.nanstd(data)),
                "trend": float(np.polyfit(range(len(data)), data, 1)[0]),
                "range": [float(np.nanmin(data)), float(np.nanmax(data))],
            }
        return result

    # Compute actual dt from times
    dt = float(np.median(np.diff(times)))
    if dt <= 0:
        dt = DT_H

    # Detrend before FFT to remove DC offset effects
    data_detrended = sp_signal.detrend(data)

    n     = len(data_detrended)
    freqs = np.fft.rfftfreq(n, d=dt)
    amps  = np.abs(np.fft.rfft(data_detrended)) * 2 / n

    # Find top 5 dominant frequencies (skip DC at index 0)
    top_idx = np.argsort(amps[1:])[::-1][:5] + 1
    dominant = []
    for idx in top_idx:
        if freqs[idx] > 0:
            dominant.append({
                "freq_per_h": float(freqs[idx]),
                "period_h":   float(1.0 / freqs[idx]),
                "amplitude":  float(amps[idx]),
            })

    result.update({
        "valid_window": [float(times[0]), float(times[-1])],
        "duration_h":   duration,
        "n_points":     n,
        "freqs":        freqs.tolist(),
        "amps":         amps.tolist(),
        "dominant":     dominant,
        "method":       "fft",
        "note":         f"FFT on valid window {times[0]:.1f}-{times[-1]:.1f}h",
    })
    return result


# ── Adaptive Hilbert transform ────────────────────────────────────────────────

def adaptive_hilbert(series: pd.Series, time_h: pd.Series) -> dict:
    """
    Hilbert transform on valid signal window only.
    Returns envelope and instantaneous phase for the valid window.
    """
    data, times, duration = get_best_window(series, time_h)

    result = {
        "signal":       series.name,
        "valid_window": None,
        "envelope":     None,
        "inst_phase":   None,
        "note":         "",
    }

    if data is None or len(data) < MIN_POINTS_WAVELET:
        result["note"] = "Insufficient data for Hilbert transform"
        return result

    analytic    = hilbert(sp_signal.detrend(data))
    envelope    = np.abs(analytic)
    inst_phase  = np.unwrap(np.angle(analytic))

    result.update({
        "valid_window": [float(times[0]), float(times[-1])],
        "time_h":       times.tolist(),
        "envelope":     envelope.tolist(),
        "inst_phase":   inst_phase.tolist(),
        "note":         f"Hilbert on {times[0]:.1f}-{times[-1]:.1f}h",
    })
    return result


# ── Adaptive CWT ──────────────────────────────────────────────────────────────

def adaptive_cwt(series: pd.Series, time_h: pd.Series,
                 wavelet: str = "morl", num_scales: int = 32) -> dict:
    """
    Continuous Wavelet Transform on valid signal window only.
    """
    data, times, duration = get_best_window(series, time_h)

    result = {
        "signal":       series.name,
        "valid_window": None,
        "scales":       None,
        "freqs":        None,
        "power":        None,
        "note":         "",
    }

    if data is None or len(data) < MIN_POINTS_WAVELET:
        result["note"] = "Insufficient data for CWT"
        return result

    dt     = float(np.median(np.diff(times)))
    if dt <= 0:
        dt = DT_H

    scales = np.geomspace(1, min(len(data)//4, 64), num=num_scales)
    coeffs, freqs = pywt.cwt(
        sp_signal.detrend(data), scales, wavelet, sampling_period=dt
    )
    power = np.abs(coeffs) ** 2

    result.update({
        "valid_window": [float(times[0]), float(times[-1])],
        "time_h":       times.tolist(),
        "scales":       scales.tolist(),
        "freqs":        freqs.tolist(),
        "power":        power.tolist(),
        "note":         f"CWT on {times[0]:.1f}-{times[-1]:.1f}h",
    })
    return result


# ── DO-specific: detect induction spike ──────────────────────────────────────

def detect_do_events(do_series: pd.Series, time_h: pd.Series,
                     min_time_h: float = 5.0) -> dict:
    """
    Detect key DO events adaptively:
    - Rapid drops (high metabolic activity)
    - Rapid spikes (carbon source depletion)
    - Pulse rhythm (methanol consumption rate)
    Works on whatever valid window exists.
    """
    data, times, duration = get_best_window(do_series, time_h)

    events = {
        "valid_window":      None,
        "induction_time_h":  None,
        "mean_do":           None,
        "do_trend":          None,
        "pulse_interval_h":  None,
        "metabolic_rate":    None,
        "note":              "",
    }

    if data is None or len(data) < 10:
        events["note"] = "Insufficient DO data"
        return events

    events["valid_window"] = [float(times[0]), float(times[-1])]
    events["mean_do"]      = float(np.mean(data))

    # Trend (positive = DO recovering, negative = consuming O2)
    trend = float(np.polyfit(range(len(data)), data, 1)[0])
    events["do_trend"]     = trend
    events["metabolic_rate"] = abs(trend)

    # Find spikes (rapid rises) after min_time_h
    dt   = float(np.median(np.diff(times)))
    ddo  = np.gradient(data)
    min_dist = max(1, int(0.5 / dt))

    peaks, _ = sp_signal.find_peaks(ddo, height=1.0, distance=min_dist)
    after_min = [p for p in peaks if times[p] >= min_time_h]

    if after_min:
        best = max(after_min, key=lambda p: ddo[p])
        events["induction_time_h"] = float(times[best])
        events["note"] = f"Induction spike at {times[best]:.1f}h"
    else:
        events["note"] = "No induction spike detected in valid window"

    # Pulse detection (for methanol phase)
    if len(after_min) > 2:
        spike_times = [times[p] for p in after_min]
        intervals   = np.diff(spike_times)
        if len(intervals) > 0:
            events["pulse_interval_h"] = float(np.median(intervals))

    return events


# ── STFT — time-frequency analysis ───────────────────────────────────────────

def rolling_stft(series: pd.Series, time_h: pd.Series,
                 window_h: float = 4.0,
                 step_h:   float = 0.5) -> dict:
    """
    Short-Time Fourier Transform: slide a window over the signal and compute
    FFT per window.  Returns a 2-D power map (time × frequency) plus a
    time-series of power at the methanol-pulse frequency (0.71 /h).

    Why 4 h window / 0.5 h step:
      - 4 h captures ~3 methanol pulses (each 1.4 h) for stable FFT
      - 0.5 h step gives smooth time resolution
    """
    data, times, duration = get_best_window(series, time_h)

    result = {
        "signal":          series.name,
        "valid_window":    None,
        "window_h":        window_h,
        "step_h":          step_h,
        "stft_times":      None,   # centre time of each window
        "stft_freqs":      None,   # frequency axis (cycles/h)
        "stft_power":      None,   # 2-D list [freq × time]
        "pulse_power":     None,   # power at METHANOL_PULSE_FREQ over time
        "pulse_freq_hz":   METHANOL_PULSE_FREQ,
        "note":            "",
    }

    if data is None or len(data) < MIN_POINTS_FFT:
        result["note"] = "Insufficient data for STFT"
        return result

    dt = float(np.median(np.diff(times)))
    if dt <= 0:
        dt = DT_H

    win_pts  = max(int(window_h / dt), MIN_POINTS_FFT)
    step_pts = max(int(step_h   / dt), 1)

    stft_times  = []
    stft_power  = []   # list of amplitude arrays, one per window
    pulse_power = []

    i = 0
    while i + win_pts <= len(data):
        chunk = data[i : i + win_pts]
        t_centre = float(times[i + win_pts // 2])

        detrended = sp_signal.detrend(chunk)
        # Hann window to reduce spectral leakage
        windowed  = detrended * np.hanning(len(detrended))
        n         = len(windowed)
        freqs     = np.fft.rfftfreq(n, d=dt)
        amps      = np.abs(np.fft.rfft(windowed)) * 2 / n
        power     = amps ** 2

        stft_times.append(t_centre)
        stft_power.append(power.tolist())

        # Extract power at methanol pulse frequency
        if len(freqs) > 1:
            freq_idx = int(np.argmin(np.abs(freqs - METHANOL_PULSE_FREQ)))
            pulse_power.append(float(power[freq_idx]))
        else:
            pulse_power.append(0.0)

        i += step_pts

    if not stft_times:
        result["note"] = "No complete windows found"
        return result

    # Frequency axis from last window (same for all windows)
    n_last = win_pts
    freqs  = np.fft.rfftfreq(n_last, d=dt)

    result.update({
        "valid_window": [float(times[0]), float(times[-1])],
        "stft_times":   stft_times,
        "stft_freqs":   freqs.tolist(),
        "stft_power":   stft_power,      # shape: [n_windows × n_freqs]
        "pulse_power":  pulse_power,     # shape: [n_windows]  ← soft OD proxy
        "note":         (f"STFT on {times[0]:.1f}-{times[-1]:.1f}h | "
                         f"{len(stft_times)} windows × {len(freqs)} freqs | "
                         f"pulse freq power tracked at {METHANOL_PULSE_FREQ:.3f}/h"),
    })
    return result


def stft_summary(stft_result: dict) -> dict:
    """
    Summarise STFT output into scalar features useful for the ML model:
      - pulse_power_mean  : mean metabolic activity during production
      - pulse_power_max   : peak methanol consumption rate
      - pulse_onset_h     : when the 0.71/h rhythm first appears (induction proxy)
      - pulse_power_slope : trend in metabolic activity (positive = ramping up)
    """
    summary = {
        "pulse_power_mean":  None,
        "pulse_power_max":   None,
        "pulse_onset_h":     None,
        "pulse_power_slope": None,
    }

    pp = stft_result.get("pulse_power")
    tt = stft_result.get("stft_times")
    if not pp or not tt:
        return summary

    pp_arr = np.array(pp)
    tt_arr = np.array(tt)

    summary["pulse_power_mean"] = float(np.mean(pp_arr))
    summary["pulse_power_max"]  = float(np.max(pp_arr))

    # Onset: first window where power exceeds 10% of max
    threshold = 0.10 * float(np.max(pp_arr))
    onset_idx = np.argmax(pp_arr > threshold)
    if pp_arr[onset_idx] > threshold:
        summary["pulse_onset_h"] = float(tt_arr[onset_idx])

    # Slope of pulse power (linear fit)
    if len(pp_arr) >= 3:
        slope = float(np.polyfit(tt_arr, pp_arr, 1)[0])
        summary["pulse_power_slope"] = slope

    return summary


# ── Stirrer as DO proxy ───────────────────────────────────────────────────────

def stirrer_as_do_proxy(stirrer: pd.Series, time_h: pd.Series) -> dict:
    """
    When DO sensor is off, use stirrer speed as a proxy.
    Stirrer increases to compensate for low DO.
    High stirrer ramp rate → low DO → high metabolic activity.
    """
    data, times, duration = get_best_window(stirrer, time_h, min_window_h=1.0)

    result = {
        "valid_window":    None,
        "ramp_rate":       None,  # rpm/h — proxy for metabolic demand
        "max_stirrer":     None,
        "stirrer_trend":   None,
        "inferred_phases": [],
        "note":            "",
    }

    if data is None or len(data) < 10:
        result["note"] = "Insufficient stirrer data"
        return result

    result["valid_window"] = [float(times[0]), float(times[-1])]
    result["max_stirrer"]  = float(np.max(data))

    # Overall trend
    trend = float(np.polyfit(range(len(data)), data, 1)[0])
    result["stirrer_trend"] = trend
    result["ramp_rate"]     = trend  # rpm per sample interval

    # Infer phases from stirrer pattern
    # Rising stirrer → DO dropping → active growth
    # Flat high stirrer → maximum aeration → high biomass
    # Falling stirrer → growth slowing → transition to production
    dt    = float(np.median(np.diff(times)))
    dstir = np.gradient(data)

    phases = []
    if trend > 0.5:
        phases.append("active_growth")
    if np.max(data) > 600:
        phases.append("high_biomass")
    if trend < -0.5 and np.max(data) > 500:
        phases.append("transition_to_production")

    result["inferred_phases"] = phases
    result["note"] = (f"Stirrer proxy on {times[0]:.1f}-{times[-1]:.1f}h "
                      f"| phases: {phases}")
    return result


# ── Full adaptive transform pipeline ─────────────────────────────────────────

def run_adaptive_transforms(df: pd.DataFrame,
                            batch_name: str = "") -> dict:
    """
    Run all transforms adaptively on whatever signals are available.
    Returns a dict of results per signal.
    """
    results = {"batch": batch_name, "signals": {}}
    t = df.time_h

    signal_configs = {
        "DO_pct":      {"fft": True, "stft": True,  "hilbert": True, "cwt": True,  "do_events": True},
        "DO_rescaled": {"fft": True, "stft": False, "hilbert": True, "cwt": False, "do_events": False},
        "pH":          {"fft": True, "stft": True,  "hilbert": True, "cwt": False, "do_events": False},
        "temp_c":      {"fft": True, "stft": False, "hilbert": False,"cwt": False, "do_events": False},
        "stirrer":     {"fft": True, "stft": False, "hilbert": False,"cwt": False, "do_events": False,
                        "proxy": True},
        "feed_pump":   {"fft": True, "stft": False, "hilbert": False,"cwt": False, "do_events": False},
    }

    for sig, config in signal_configs.items():
        if sig not in df.columns:
            continue

        series  = df[sig]
        nn      = series.notna().sum()
        if nn == 0:
            continue

        sig_result = {"n_valid": int(nn)}

        if config.get("fft"):
            sig_result["fft"] = adaptive_fft(series, t, label=sig)
        if config.get("stft"):
            sig_result["stft"] = rolling_stft(series, t)
            sig_result["stft_summary"] = stft_summary(sig_result["stft"])
        if config.get("hilbert"):
            sig_result["hilbert"] = adaptive_hilbert(series, t)
        if config.get("cwt"):
            sig_result["cwt"] = adaptive_cwt(series, t)
        if config.get("do_events"):
            sig_result["do_events"] = detect_do_events(series, t)
        if config.get("proxy"):
            sig_result["proxy"] = stirrer_as_do_proxy(series, t)

        results["signals"][sig] = sig_result

    return results


def summarise_transforms(results: dict):
    """Print a human-readable summary of transform results."""
    print(f"\n{'='*60}")
    print(f"  Adaptive Transform Summary: {results['batch']}")
    print(f"{'='*60}")

    for sig, data in results["signals"].items():
        print(f"\n── {sig} ({data['n_valid']} valid points) ──")

        if "fft" in data:
            fft = data["fft"]
            print(f"  FFT: {fft['note']}")
            if fft.get("dominant"):
                for d in fft["dominant"][:3]:
                    print(f"    period={d['period_h']:.2f}h  "
                          f"amp={d['amplitude']:.3f}")
            if fft.get("stats"):
                s = fft["stats"]
                print(f"  Stats: mean={s['mean']:.2f}  "
                      f"std={s['std']:.2f}  trend={s['trend']:.4f}/sample")

        if "stft" in data:
            st = data["stft"]
            sm = data.get("stft_summary", {})
            print(f"  STFT: {st['note']}")
            if sm.get("pulse_onset_h") is not None:
                print(f"    pulse onset (induction proxy): {sm['pulse_onset_h']:.1f}h")
            if sm.get("pulse_power_max") is not None:
                print(f"    max metabolic power @ {METHANOL_PULSE_FREQ:.2f}/h: "
                      f"{sm['pulse_power_max']:.4f}")
            if sm.get("pulse_power_slope") is not None:
                trend = "↑ ramping" if sm["pulse_power_slope"] > 0 else "↓ declining"
                print(f"    metabolic activity trend: {trend} "
                      f"({sm['pulse_power_slope']:.5f}/h)")

        if "do_events" in data:
            ev = data["do_events"]
            print(f"  DO events: {ev['note']}")
            if ev.get("induction_time_h"):
                print(f"    induction spike: {ev['induction_time_h']:.1f}h")
            if ev.get("pulse_interval_h"):
                print(f"    pulse interval:  {ev['pulse_interval_h']:.2f}h")

        if "proxy" in data:
            px = data["proxy"]
            print(f"  Stirrer proxy: {px['note']}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from loader import load_batch
    from pathlib import Path

    for f in sorted(Path("data").glob("*.xlsx")):
        print(f"\nLoading {f.name}...")
        sheets = load_batch(f)
        df = list(sheets.values())[0]
        run = df[df.phase == "run"].copy()

        # Auto-detect experiment end
        diffs = run.time_h.diff().dropna()
        large = diffs[diffs > 1.0]
        if not large.empty:
            end_h = float(run.loc[large.index[0] - 1, "time_h"])
            run   = run[run.time_h <= end_h]

        results = run_adaptive_transforms(run, batch_name=f.stem)
        summarise_transforms(results)
