import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, "src")


def compute_growth_rate(sparse_df: pd.DataFrame,
                        od_col: str = "OD660",
                        dcw_col: str = "DCW_gL") -> pd.DataFrame:
    """
    Compute specific growth rate (mu, h-1) from sparse OD660 or DCW.
    mu = d(ln X)/dt between consecutive timepoints.
    """
    df = sparse_df.dropna(subset=[dcw_col if dcw_col in sparse_df.columns
                                   else od_col]).copy()
    col = dcw_col if dcw_col in df.columns else od_col
    df = df[["time_h", col]].dropna().sort_values("time_h").reset_index(drop=True)

    mus = [np.nan]
    for i in range(1, len(df)):
        dt  = df.loc[i, "time_h"] - df.loc[i-1, "time_h"]
        x1  = df.loc[i-1, col]
        x2  = df.loc[i,   col]
        if dt > 0 and x1 > 0 and x2 > 0:
            mus.append(np.log(x2 / x1) / dt)
        else:
            mus.append(np.nan)
    df["mu_per_h"] = mus
    return df


def detect_phases(sparse_df: pd.DataFrame) -> dict:
    """
    Detect key cultivation phases from sparse measurements.
    Returns dict with timepoints of key events.
    """
    events = {}
    df = sparse_df.copy()

    # Glycerol depletion — first time glycerol drops below 1 g/L
    if "glycerol_gL" in df.columns:
        depleted = df[df.glycerol_gL < 1.0]
        if not depleted.empty:
            events["glycerol_depletion_h"] = float(depleted.time_h.iloc[0])

    # Methanol induction start — first time methanol > 0
    if "methanol_gL" in df.columns:
        induced = df[df.methanol_gL > 0.1]
        if not induced.empty:
            events["induction_start_h"] = float(induced.time_h.iloc[0])

    # Peak L1 yield timepoint
    if "L1_yield_mgL" in df.columns:
        peak_idx = df.L1_yield_mgL.idxmax()
        if pd.notna(peak_idx):
            events["peak_L1_h"]    = float(df.loc[peak_idx, "time_h"])
            events["peak_L1_mgL"]  = float(df.loc[peak_idx, "L1_yield_mgL"])

    # Max OD660
    if "OD660" in df.columns:
        max_idx = df.OD660.idxmax()
        if pd.notna(max_idx):
            events["max_OD660_h"]  = float(df.loc[max_idx, "time_h"])
            events["max_OD660"]    = float(df.loc[max_idx, "OD660"])

    # Max DCW
    if "DCW_gL" in df.columns:
        max_idx = df.DCW_gL.idxmax()
        if pd.notna(max_idx):
            events["max_DCW_h"]    = float(df.loc[max_idx, "time_h"])
            events["max_DCW_gL"]   = float(df.loc[max_idx, "DCW_gL"])

    return events


def continuous_features(cont_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract rolling features from continuous signals.
    These will feed into the real-time controller.
    """
    df = cont_df.copy().sort_values("time_h").reset_index(drop=True)

    for sig in ["DO_pct", "pH", "stirrer", "temp_c"]:
        if sig not in df.columns:
            continue
        s = df[sig]
        # Rolling mean and std over 1h window (12 samples at 5-min intervals)
        df[f"{sig}_mean1h"] = s.rolling(12, min_periods=1).mean()
        df[f"{sig}_std1h"]  = s.rolling(12, min_periods=1).std()
        # Rate of change (derivative)
        df[f"{sig}_ddt"]    = s.diff() / df.time_h.diff()

    # DO drop rate — key indicator of microbial activity
    if "DO_pct" in df.columns:
        df["DO_drop_rate"] = -df["DO_pct_ddt"].clip(upper=0)

    # pH drift from start
    if "pH" in df.columns:
        df["pH_drift"] = df["pH"] - df["pH"].iloc[0]

    # Stirrer ramp — indicates DO compensation
    if "stirrer" in df.columns:
        df["stirrer_ramp"] = df["stirrer_ddt"].clip(lower=0)

    return df


def summarise(batch_name: str, sparse_df: pd.DataFrame,
              cont_df: pd.DataFrame = None):
    """Print a full feature summary for a batch."""
    print(f"\n{'='*50}")
    print(f"  {batch_name}")
    print(f"{'='*50}")

    # Phase events
    events = detect_phases(sparse_df)
    print("\nKey events:")
    for k, v in events.items():
        print(f"  {k:30s}: {v:.2f}")

    # Growth rates
    gr = compute_growth_rate(sparse_df)
    print(f"\nGrowth rate (mu):")
    print(gr[["time_h", "DCW_gL" if "DCW_gL" in gr.columns else "OD660",
              "mu_per_h"]].to_string(index=False))

    # Continuous features summary
    if cont_df is not None:
        cf = continuous_features(cont_df)
        print(f"\nContinuous signal summary ({len(cf)} rows):")
        for col in ["DO_pct", "pH", "stirrer"]:
            if col in cf.columns:
                print(f"  {col:12s}: mean={cf[col].mean():.2f}, "
                      f"std={cf[col].std():.2f}, "
                      f"max_ddt={cf[f'{col}_ddt'].abs().max():.3f}")


if __name__ == "__main__":
    from loader import load_batch, load_sparse_samples

    for fname, bname in [("data/result-batch_5.xlsx", "Batch 5"),
                         ("data/result-batch_7.xlsx", "Batch 7")]:
        sparse = load_sparse_samples(fname)
        sheets = load_batch(fname)
        cont   = list(sheets.values())[0]
        cont   = cont[cont.phase == "run"].copy()

        if sparse is not None:
            summarise(bname, sparse, cont)
