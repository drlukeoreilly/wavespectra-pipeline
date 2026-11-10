# 01_ws_waterfall_ef.py
from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import wavespectra as ws  # ensure .spec accessor is registered

# --- CONFIG ---
CSV_PATH = Path(r"C:\Users\LukeOReilly\OneDrive - GREEN REBEL GROUP\Desktop\IDS\Vattenfall\May 2025 Submission\03_QCed\03_DWR_FLS\Test.csv")
DAYFIRST = True
PNG_DPI = 150
MAX_X_TICKS = 8   # adjust if you want more/less ticks
DATE_FMT = "%Y-%m-%d"  # or "%d-%b-%Y"
SCRIPT_NAME = "01_ws_waterfall_ef"
# -------------

def format_date_axis(ax):
    """Make datetime x-axis readable (no overlap)."""
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=MAX_X_TICKS))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(DATE_FMT))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.gcf().autofmt_xdate()  # extra safeguard

def load_ef(csv_path: Path) -> xr.DataArray:
    df = pd.read_csv(csv_path)
    time = pd.to_datetime(df.get("timestamp"), dayfirst=DAYFIRST, errors="coerce")

    pat = re.compile(r"^(\d+(?:\.\d+)?)Hz_SpectralDensity_M2pHz$")  # match frequency columns
    cols, f = [], []
    for c in df.columns:
        m = pat.match(str(c).strip())
        if m:
            cols.append(c)
            f.append(float(m.group(1)))
    if not cols:
        raise ValueError("No '<freq>Hz_SpectralDensity_M2pHz' columns found.")

    order = np.argsort(f)
    f = np.array(f)[order]
    ef = df[cols].to_numpy(dtype=float)[:, order]

    da = xr.DataArray(
        ef, coords={"time": time, "freq": f}, dims=("time", "freq"),
        name="ef", attrs={"units": "m^2/Hz"}
    )
    return da

def main():
    tqdm.write(f"Reading {CSV_PATH}")
    da = load_ef(CSV_PATH)

    # Create output directory
    out_dir = CSV_PATH.parent / SCRIPT_NAME
    out_dir.mkdir(exist_ok=True)

    # Peak frequency per time (skip empty rows)
    peak_f = np.full(da.sizes["time"], np.nan)
    for i in tqdm(range(da.sizes["time"]), desc="Finding peak frequency"):
        y = da.isel(time=i).values
        if y.size and not (np.all(np.isnan(y)) or np.nanmax(np.abs(y)) == 0):
            peak_f[i] = da["freq"].values[np.nanargmax(y)]

    out_csv = out_dir / f"{CSV_PATH.stem}_PeakFrequency.csv"
    pd.DataFrame({
        "time": pd.to_datetime(da.time.values),
        "f_peak_Hz": peak_f,
        "Tp_s": np.where(~np.isnan(peak_f), 1/peak_f, np.nan)
    }).to_csv(out_csv, index=False)
    tqdm.write(f"Saved: {out_csv}")

    # Waterfall (E(f) vs time)
    valid = np.array([
        not (np.all(np.isnan(da.isel(time=i).values)) or np.nanmax(np.abs(da.isel(time=i).values)) == 0)
        for i in range(da.sizes["time"])
    ])
    if valid.any():
        sub = da.isel(time=np.where(valid)[0])
        fig, ax = plt.subplots(figsize=(12, 5))
        mesh = ax.pcolormesh(pd.to_datetime(sub["time"].values),
                             sub["freq"].values, sub.T, shading="auto")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"{CSV_PATH.stem} — Time–Frequency Energy (ef)")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("m²/Hz")

        # tidy date axis
        format_date_axis(ax)

        out_png = out_dir / f"{CSV_PATH.stem}_waterfall_ef.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=PNG_DPI)
        plt.close(fig)
        tqdm.write(f"Saved: {out_png}")
    else:
        tqdm.write("No usable spectra to plot.")

    tqdm.write("Done ✅")

if __name__ == "__main__":
    main()
