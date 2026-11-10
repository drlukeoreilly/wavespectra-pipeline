# 03_ws_export_to_netcdf.py
from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr

# --- CONFIG ---
CSV_PATH = Path(r"C:\Users\LukeOReilly\OneDrive - GREEN REBEL GROUP\Desktop\IDS\Vattenfall\May 2025 Submission\03_QCed\03_DWR_FLS\Test.csv")
DAYFIRST = True
# -------------

def load_ef(csv_path: Path) -> xr.Dataset:
    """Parse Vattenfall-style CSV into an xarray Dataset with ef(time,freq)."""
    df = pd.read_csv(csv_path)

    # Time axis
    time = pd.to_datetime(df.get("timestamp"), dayfirst=DAYFIRST, errors="coerce")

    # Identify per-frequency spectral density columns: '<freq>Hz_SpectralDensity_M2pHz'
    pat = re.compile(r"^(\d+(?:\.\d+)?)Hz_SpectralDensity_M2pHz$")
    ef_cols, freqs = [], []
    for c in df.columns:
        s = str(c).strip()
        m = pat.match(s)
        if m:
            ef_cols.append(c)
            freqs.append(float(m.group(1)))
    if not ef_cols:
        raise ValueError("No ef columns found (expected pattern '<freq>Hz_SpectralDensity_M2pHz').")

    # Sort by frequency
    order = np.argsort(freqs)
    f = np.array(freqs, dtype=float)[order]
    ef = df[ef_cols].to_numpy(dtype=float)[:, order]  # (time, freq)

    # Build Dataset
    ds = xr.Dataset(
        data_vars=dict(
            ef=(("time", "freq"), ef, {"units": "m^2/Hz", "long_name": "Omnidirectional spectral density"})
        ),
        coords=dict(
            time=("time", time, {"standard_name": "time"}),
            freq=("freq", f, {"units": "Hz", "long_name": "Frequency"})
        ),
        attrs=dict(
            title="Omnidirectional wave spectra (from DWR CSV)",
            source=str(csv_path),
            history="Created by 03_ws_export_to_netcdf.py",
            conventions="CF-inspired (not strict CF)",
            note="Parsed from per-frequency columns '<freq>Hz_SpectralDensity_M2pHz'.",
        ),
    )

    # If useful bulk columns exist, preserve them as auxiliary vars
    for col in ["Hm0_m", "Hs_m", "Tp_s", "Tz_s", "PeakDir_deg", "MeanDir_deg", "MeanSpread_deg"]:
        if col in df.columns:
            # Try to coerce to float series where possible
            s = pd.to_numeric(df[col], errors="coerce")
            ds[col] = (("time",), s)

    return ds

def main():
    ds = load_ef(CSV_PATH)

    # Set compression/chunking for efficient storage
    # (float32 to reduce size, zlib compression; chunk along time to stream better on reload)
    encoding = {
        "ef": {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (min(len(ds.time), 256), len(ds.freq))},
        # coords usually small, but be explicit about dtype
        "time": {"dtype": "datetime64[ns]"},
        "freq": {"dtype": "float32"},
    }

    out_nc = CSV_PATH.with_suffix(".nc")
    ds.to_netcdf(out_nc, encoding=encoding)

    print("Saved NetCDF:", out_nc)
    print("Variables:", list(ds.data_vars))
    print("Dims:", dict(ds.dims))

if __name__ == "__main__":
    main()
