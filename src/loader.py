import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def _parse_bioreactor_sheet(df_raw):
    """Extract clean time-series from a messy bioreactor Excel sheet."""
    header_row = None
    for i, row in df_raw.iterrows():
        vals = row.fillna("").astype(str).str.lower().tolist()
        if any("date" in v for v in vals) and any("time" in v for v in vals):
            header_row = i
            break

    if header_row is None:
        raise ValueError("Could not find header row with 'Date' and 'Time'")

    col_labels = df_raw.iloc[header_row].fillna("").astype(str).str.strip().tolist()
    data = df_raw.iloc[header_row + 1:].copy().reset_index(drop=True)

    valid_mask = pd.to_datetime(data.iloc[:, 0], errors="coerce").notna()
    data = data[valid_mask].reset_index(drop=True)

    date_str = data.iloc[:, 0].astype(str).str.strip()
    time_str = data.iloc[:, 1].astype(str).str.strip()
    data["datetime"] = pd.to_datetime(date_str + " " + time_str, format="mixed", errors="coerce")
    data = data.dropna(subset=["datetime"]).reset_index(drop=True)

    elapsed_col_idx = None
    for idx, label in enumerate(col_labels):
        if label.lower() in ("time", "time (h)", "time(h)", "hour", "hours"):
            elapsed_col_idx = idx
            break

    if elapsed_col_idx is not None:
        time_h = pd.to_numeric(data.iloc[:, elapsed_col_idx], errors="coerce").copy()
        valid_t = time_h.notna()
        if valid_t.any():
            t0_dt = data.loc[valid_t, "datetime"].iloc[0]
            t0_h  = float(time_h[valid_t].iloc[0])
        else:
            t0_dt = data["datetime"].iloc[0]
            t0_h  = 0.0
        fill_mask = time_h.isna()
        time_h.loc[fill_mask] = (
            (data.loc[fill_mask, "datetime"] - t0_dt).dt.total_seconds() / 3600 + t0_h
        )
        data["time_h"] = time_h.values
    else:
        t0 = data["datetime"].iloc[0]
        data["time_h"] = (data["datetime"] - t0).dt.total_seconds() / 3600

    signal_map = {
        "stirrer":      ["stirrer", "stir", "rpm"],
        "temp_c":       ["temp", "temperature"],
        "pH":           ["ph"],
        "DO_pct":       ["%do", "dissolved"],
        "DO_rescaled":  ["rescale", "rescaled"],
        "acid_pump":    ["acid"],
        "base_pump":    ["base"],
        "feed_pump":    ["feed"],
        "OD660":        ["od660", "od_660"],
        "glycerol_gL":  ["glycerol"],
        "methanol_gL":  ["meoh", "methanol"],
        "DCW_gL":       ["dcw"],
        "L1_yield_mgL": ["l1 yield", "l1yield", "l1_yield"],
    }

    clean = pd.DataFrame({"datetime": data["datetime"], "time_h": data["time_h"]})

    for target, keywords in signal_map.items():
        matched_idx = None
        for idx, label in enumerate(col_labels):
            ls = label.lower()
            if any(k in ls for k in keywords):
                matched_idx = idx
                break
        if matched_idx is not None and matched_idx < data.shape[1]:
            clean[target] = pd.to_numeric(data.iloc[:, matched_idx], errors="coerce")

    clean = clean.sort_values("time_h").reset_index(drop=True)
    # Tag pre-inoculation rows (negative time) for easy filtering
    clean["phase"] = clean["time_h"].apply(lambda t: "pre" if t < 0 else "run")
    return clean


def load_batch(filepath):
    filepath = Path(filepath)
    xl = pd.ExcelFile(filepath)
    results = {}

    for sheet in xl.sheet_names:
        raw = xl.parse(sheet, header=None)
        flat = raw.fillna("").astype(str).values.flatten()
        has_date = any("date" in v.lower() for v in flat)
        has_do   = any("%do" in v.lower() or "dissolved" in v.lower() for v in flat)
        if has_date and has_do:
            try:
                clean = _parse_bioreactor_sheet(raw)
                results[sheet] = clean
                print(f"  Loaded '{sheet}': {len(clean)} rows, "
                      f"{clean['time_h'].min():.1f}-{clean['time_h'].max():.1f} h, "
                      f"cols: {[c for c in clean.columns if c not in ('datetime',)]}")
            except Exception as e:
                print(f"  Skipped '{sheet}': {e}")

    return results


def load_all(data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)
    batches = {}
    for f in sorted(data_dir.glob("*.xlsx")):
        print(f"\nLoading {f.name} ...")
        batches[f.stem] = load_batch(f)
    return batches


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    batches = load_all(data_dir)
    for bname, sheets in batches.items():
        for sname, df in sheets.items():
            print(f"\n=== {bname} / {sname} ===")
            print(df.head(8).to_string())


def load_sparse_samples(filepath) -> pd.DataFrame | None:
    """
    Load manually sampled data (OD660, DCW, glycerol, methanol, L1 yield)
    from the lab notebook sheet — covers the full cultivation run.
    """
    filepath = Path(filepath)
    xl = pd.ExcelFile(filepath)

    # Look for sheets with Time(h) + L1 Yield columns
    for sheet in xl.sheet_names:
        raw = xl.parse(sheet, header=None)
        flat = raw.fillna("").astype(str).values.flatten()
        has_time   = any("time" in v.lower() and "h" in v.lower() for v in flat)
        has_l1     = any("l1" in v.lower() for v in flat)
        has_od     = any("od660" in v.lower() or "od 660" in v.lower() for v in flat)
        if has_time and (has_l1 or has_od):
            # Find header row
            for i, row in raw.iterrows():
                vals = row.fillna("").astype(str).str.lower().tolist()
                if any("time" in v for v in vals) and any("od" in v for v in vals):
                    cols = raw.iloc[i].fillna("").astype(str).str.strip().tolist()
                    data = raw.iloc[i+1:].copy().reset_index(drop=True)

                    # Map columns
                    col_map = {}
                    for idx, label in enumerate(cols):
                        ll = label.lower()
                        if "time" in ll:               col_map["time_h"]       = idx
                        elif "od660" in ll or "od" in ll: col_map["OD660"]     = idx
                        elif "dcw" in ll:              col_map["DCW_gL"]       = idx
                        elif "glycerol" in ll:         col_map["glycerol_gL"]  = idx
                        elif "meoh" in ll or "methanol" in ll: col_map["methanol_gL"] = idx
                        elif "l1" in ll:               col_map["L1_yield_mgL"] = idx

                    if "time_h" not in col_map:
                        continue

                    result = pd.DataFrame()
                    for target, cidx in col_map.items():
                        result[target] = pd.to_numeric(
                            data.iloc[:, cidx], errors="coerce"
                        )

                    result = result.dropna(subset=["time_h"]).reset_index(drop=True)
                    result = result.sort_values("time_h").reset_index(drop=True)
                    print(f"  Sparse sheet '{sheet}': {len(result)} rows, "
                          f"cols: {list(result.columns)}")
                    return result

    return None
