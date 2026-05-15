import pandas as pd
import numpy as np
from pathlib import Path

DEFAULT_MAX_H = 200

SIGNAL_LIMITS = {
    "DO_pct":       (-20,  110),
    "DO_rescaled":  (-5,   110),
    "pH":           (3.0,  8.0),
    "temp_c":       (20,   45),
    "stirrer":      (0,    1500),
    "acid_pump":    (0,    10000),
    "feed_pump":    (0,    100000),
    "OD660":        (0,    300),
    "glycerol_gL":  (0,    500),
    "methanol_gL":  (0,    100),
    "DCW_gL":       (0,    200),
    "L1_yield_mgL": (0,    5000),
}

CONTINUOUS = ["DO_pct", "DO_rescaled", "pH", "temp_c",
              "stirrer", "acid_pump", "feed_pump"]
SPARSE     = ["OD660", "glycerol_gL", "methanol_gL", "DCW_gL", "L1_yield_mgL"]


def detect_experiment_end(df: pd.DataFrame) -> float:
    """
    Auto-detect when the experiment actually ended.
    Strategy: find where continuous signals stop changing (flatline)
    or where the logger disconnected (large time gap).
    """
    run = df[df.phase == "run"].copy()
    if run.empty:
        return DEFAULT_MAX_H

    # Find largest gap in time — logger likely disconnected there
    time_diffs = run.time_h.diff().dropna()
    if time_diffs.empty:
        return float(run.time_h.max())

    # If max gap > 1h, experiment likely ended before that gap
    large_gaps = time_diffs[time_diffs > 1.0]
    if not large_gaps.empty:
        # End is just before the first large gap
        gap_idx  = large_gaps.index[0]
        end_time = float(run.loc[gap_idx - 1, "time_h"]) \
                   if gap_idx > 0 else float(run.time_h.max())
        return end_time

    return float(run.time_h.max())


def clean(df: pd.DataFrame, batch_name: str = "",
          max_h: float = None) -> pd.DataFrame:
    """Clean and preprocess a batch DataFrame."""

    # 1 — run phase only
    df = df[df.phase == "run"].copy()

    # 2 — auto-detect or use provided max_h
    if max_h is None:
        max_h = detect_experiment_end(df)
        print(f"  Auto-detected experiment end: {max_h:.1f} h")

    df = df[df.time_h <= max_h].copy()

    # 3 — spike removal
    for sig, (lo, hi) in SIGNAL_LIMITS.items():
        if sig in df.columns:
            mask = (df[sig] < lo) | (df[sig] > hi)
            if mask.any():
                print(f"  [{sig}] removed {mask.sum()} spikes")
            df.loc[mask, sig] = np.nan

    # 4 — sort
    df = df.sort_values("time_h").reset_index(drop=True)

    # 5 — interpolate continuous signals only
    for sig in CONTINUOUS:
        if sig in df.columns:
            df[sig] = df[sig].interpolate(method="index", limit=6)

    # 6 — tag sparse rows
    available_sparse = [s for s in SPARSE if s in df.columns]
    df["has_sparse"] = df[available_sparse].notna().any(axis=1) \
                       if available_sparse else False

    # 7 — signal availability report
    print(f"  Clean: {len(df)} rows | time: {df.time_h.min():.1f}-{df.time_h.max():.1f} h")
    for sig in CONTINUOUS:
        if sig in df.columns:
            nn = df[sig].notna().sum()
            print(f"    {sig:15s}: {nn}/{len(df)} continuous")
    for sig in SPARSE:
        if sig in df.columns:
            nn = df[sig].notna().sum()
            if nn > 0:
                print(f"    {sig:15s}: {nn} sparse points")

    return df


def process_all(data_dir: str = "data",
                out_dir:  str = "outputs/processed_data"):
    """Process all xlsx files found in data_dir."""
    import sys
    sys.path.insert(0, "src")
    from loader import load_batch

    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for f in sorted(data_dir.glob("*.xlsx")):
        batch_name = f.stem
        print(f"\nProcessing {f.name} ...")
        try:
            sheets = load_batch(f)
            for sheet_name, df in sheets.items():
                df_clean = clean(df, batch_name=batch_name)
                out_path = out_dir / f"{batch_name}.csv"
                df_clean.to_csv(out_path, index=False)
                print(f"  Saved -> {out_path}")
                results[batch_name] = df_clean
        except Exception as e:
            print(f"  Error processing {f.name}: {e}")

    return results


if __name__ == "__main__":
    process_all()
