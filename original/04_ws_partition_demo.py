# 04_ws_partition_demo.py
# PTM1_TRACK time series (wind-sea + swells) + PTM1_TRACK polar partition snapshot.
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import re
import wavespectra as ws  # registers .spec accessor

# ---------------- CONFIG ----------------
DATA_DIR     = Path(r"C:\Users\LukeOReilly\OneDrive - GREEN REBEL GROUP\Desktop\IDS\Vattenfall\May 2025 Submission\03_QCed\03_DWR_FLS")
SPECTRA_FILE = "Test.csv"
WIND_FILE    = "u and v_speed_dir.csv"
DEPTH_M      = 71.0
TOP_SWELLS   = 4
PNG_DPI      = 150
OUT_FOLDER   = "04_ws_partition_demo"   # new subfolder for all outputs
# ---------------------------------------

def format_date_axis(ax, fmt="%Y-%m-%d", maxticks=10, rot=30):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=maxticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    plt.setp(ax.get_xticklabels(), rotation=rot, ha="right")
    plt.gcf().autofmt_xdate()

def load_perfreq_columns(df: pd.DataFrame):
    pat_ef  = re.compile(r"^(\d+(?:\.\d+)?)Hz_SpectralDensity_M2pHz$")
    pat_dir = re.compile(r"^(\d+(?:\.\d+)?)Hz_DirectionFrom_deg$")
    pat_spr = re.compile(r"^(\d+(?:\.\d+)?)Hz_Spread_deg$")

    ef_cols, freqs = [], []
    dir_cols, spr_cols = {}, {}
    for c in df.columns:
        s = str(c).strip()
        m = pat_ef.match(s)
        if m:
            ef_cols.append(c); freqs.append(float(m.group(1)))
        m2 = pat_dir.match(s)
        if m2:
            dir_cols[float(m2.group(1))] = c
        m3 = pat_spr.match(s)
        if m3:
            spr_cols[float(m3.group(1))] = c

    if not ef_cols:
        raise ValueError("No ef columns found.")

    f = np.sort(np.array(freqs))
    return f, ef_cols, dir_cols, spr_cols

def build_efth(df, time, f, ef_cols, dir_cols, spr_cols, theta_bins=72):
    thetas = np.linspace(0, 360, theta_bins, endpoint=False)
    efth = np.zeros((len(time), len(f), len(thetas)), dtype=float)

    def f_from_col(c):
        return float(re.match(r"^(\d+(?:\.\d+)?)", c).group(1))

    def circ_gauss(tdeg, mu, sig):
        sig = max(float(sig), 1e-3)
        d = np.minimum(np.mod(tdeg - mu, 360), np.mod(mu - tdeg, 360))
        return np.exp(-(d ** 2) / (2 * sig ** 2))

    for it in tqdm(range(len(time)), desc="Constructing E(f,θ)"):
        row = df.iloc[it]
        ef_row = np.array([row[c] for c in ef_cols], dtype=float)
        if ef_row.size == 0 or np.all(np.isnan(ef_row)) or np.nanmax(np.abs(ef_row)) == 0:
            continue
        th = np.array([row[dir_cols.get(f_from_col(c), np.nan)] for c in ef_cols], dtype=float)
        sp = np.array([row[spr_cols.get(f_from_col(c), np.nan)] for c in ef_cols], dtype=float)

        if (not np.isfinite(th).any()) and ("MeanDir_deg" in df.columns):
            th[:] = float(row["MeanDir_deg"])
        if (not np.isfinite(sp).any()) and ("MeanSpread_deg" in df.columns):
            sp[:] = max(float(row["MeanSpread_deg"]), 5.0)

        for i, fi in enumerate(f):
            efi = ef_row[i]
            if not (np.isfinite(efi) and efi > 0 and np.isfinite(th[i]) and np.isfinite(sp[i])):
                continue
            w = circ_gauss(thetas, th[i], sp[i])
            sw = w.sum()
            if sw > 0:
                w /= sw
            efth[it, i, :] = efi * w

    return xr.DataArray(
        efth,
        coords={"time": time, "freq": f, "dir": thetas},
        dims=("time", "freq", "dir"),
        name="efth",
        attrs={"units": "m^2/Hz/deg"}
    )

def interpolate_wind_to_wave_times(wind_df, wave_time):
    w = wind_df.copy()
    w["timestamp"] = pd.to_datetime(w["timestamp"], dayfirst=True, errors="coerce")
    w = w.sort_values("timestamp").dropna(subset=["timestamp"])
    spd = pd.to_numeric(w["wind_speed"], errors="coerce")
    wdir = pd.to_numeric(w["wind_direction"], errors="coerce")
    # Interpolate via vector components to respect circular nature of direction
    u = spd * np.sin(np.deg2rad(wdir))
    v = spd * np.cos(np.deg2rad(wdir))
    wind_vec = pd.DataFrame({"timestamp": w["timestamp"], "u": u, "v": v}).dropna()
    idx = pd.DataFrame({"timestamp": pd.to_datetime(wave_time)})

    mu = pd.merge_asof(idx.sort_values("timestamp"),
                       wind_vec[["timestamp", "u"]].sort_values("timestamp"),
                       on="timestamp", direction="nearest", tolerance=pd.Timedelta("2H"))
    mv = pd.merge_asof(idx.sort_values("timestamp"),
                       wind_vec[["timestamp", "v"]].sort_values("timestamp"),
                       on="timestamp", direction="nearest", tolerance=pd.Timedelta("2H"))
    uv = pd.DataFrame({"timestamp": idx["timestamp"], "u": mu["u"], "v": mv["v"]})
    uv["u"] = uv["u"].interpolate(limit_direction="both")
    uv["v"] = uv["v"].interpolate(limit_direction="both")

    speed = np.hypot(uv["u"], uv["v"])
    direction = (np.rad2deg(np.arctan2(uv["u"], uv["v"])) + 360) % 360
    return pd.DataFrame({"timestamp": idx["timestamp"], "wind_speed": speed, "wind_direction": direction})

def plot_windsea_swell(dspart, title, out_dir: Path, stem: str):
    # Stats dataframe
    df = dspart.spec.stats(["hs", "tp", "dpm"]).to_dataframe().reset_index()
    df = df[df["hs"] > 0]

    # Identify wind-sea partition (part=0) and top swells
    ws_df = df[df["part"] == 0]
    swell_df = df[df["part"] >= 1]
    swell_energy = swell_df.groupby("part")["hs"].apply(lambda x: (x**2).sum()).sort_values(ascending=False)
    top_swells = swell_energy.head(TOP_SWELLS).index.tolist()

    # Keep only wind-sea + top swells; group the rest
    df_sel = pd.concat([ws_df, swell_df[swell_df["part"].isin(top_swells)]])
    others = swell_df[~swell_df["part"].isin(top_swells)].copy()
    if not others.empty:
        others["part"] = -1  # "Others"
        df_sel = pd.concat([df_sel, others])

    # Plot Hs time series
    plt.figure(figsize=(11, 5))
    for pid, grp in df_sel.groupby("part"):
        if pid == 0:
            label = "Wind-sea"
        elif pid == -1:
            label = "Others"
        else:
            label = f"Swell {pid}"
        plt.plot(grp["time"], grp["hs"], marker="o", ms=2, lw=1, label=label)

    plt.ylabel("Hs (m)")
    plt.xlabel("Time")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    format_date_axis(plt.gca())
    plt.tight_layout()

    save_path = out_dir / f"{stem}_PTM1_TRACK_Hs.png"
    plt.savefig(save_path, dpi=PNG_DPI)
    print(f"Saved figure to: {save_path}")
    plt.close()

def main():
    # --- Output directory ---
    out_dir = DATA_DIR / OUT_FOLDER
    out_dir.mkdir(exist_ok=True)

    # Load spectra CSV
    spec_path = DATA_DIR / SPECTRA_FILE
    df = pd.read_csv(spec_path)
    time = pd.to_datetime(df["timestamp"], dayfirst=True)

    # Load wind and align
    wind_df = pd.read_csv(DATA_DIR / WIND_FILE)
    wind_interp = interpolate_wind_to_wave_times(wind_df, time)

    # Build efth and dataset
    f, ef_cols, dir_cols, spr_cols = load_perfreq_columns(df)
    efth = build_efth(df, time, f, ef_cols, dir_cols, spr_cols)
    dset = efth.to_dataset()
    dset["wspd"] = (("time",), wind_interp["wind_speed"].to_numpy())
    dset["wdir"] = (("time",), wind_interp["wind_direction"].to_numpy())
    dset["dpt"]  = (("time",), np.full(len(time), DEPTH_M))

    stem = spec_path.stem

    # ---------- PTM1_TRACK over time (wind-sea + swells) ----------
    try:
        dspart = dset.spec.partition.ptm1_track(
            wspd=dset.wspd,
            wdir=dset.wdir,
            dpt=dset.dpt,
        )
        # Save CSV of hs/tp/dpm
        csv_path = out_dir / f"{stem}_TRACKED_PTM1_timeseries.csv"
        dspart.spec.stats(["hs", "tp", "dpm"]).to_dataframe().reset_index().to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

        # Plot wind-sea + top swells Hs
        plot_windsea_swell(dspart, "PTM1_TRACK — Wind-sea + Top Swells (Hs)", out_dir, stem)

        # ---------- PTM1_TRACK polar partition snapshot (small multiples) ----------
        # Pick the first valid time step
        ok_idx = None
        for i in range(dspart.sizes["time"]):
            # sum energy to detect non-empty
            arr = dspart["efth"].isel(time=i).values if "efth" in dspart else None
            if arr is not None and np.isfinite(arr).any() and np.nanmax(arr) > 0:
                ok_idx = i
                break
        if ok_idx is not None:
            snap = dspart.isel(time=ok_idx, drop=True)
            # Choose a column dimension for facets (prefer 'part', else 'npart_id'/'part_id')
            col_dim = "part"
            for cand in ["part", "npart_id", "part_id"]:
                if cand in snap.dims:
                    col_dim = cand
                    break
            try:
                g = snap.spec.plot(col=col_dim)  # polar panels per partition
                # Clean titles a bit
                for ax in np.ravel(g.axs):
                    ax.set_title(ax.get_title().replace(f"{col_dim} = ", "Partition "))
                ts = pd.to_datetime(time.iloc[ok_idx]).strftime("%Y%m%d_%H%M")
                out_png = out_dir / f"{stem}_PTM1_TRACK_partition_snapshot_{ts}.png"
                plt.savefig(out_png, dpi=PNG_DPI, bbox_inches="tight")
                plt.close()
                print(f"Saved PTM1_TRACK snapshot plot: {out_png}")
            except Exception as e:
                print("PTM1_TRACK snapshot plotting failed (skipped). Reason:", e)
        else:
            print("No usable spectrum found for PTM1_TRACK snapshot; skipped.")

    except Exception as e:
        print("PTM1_TRACK failed (skipped). Reason:", e)

    print("Done ✅")

if __name__ == "__main__":
    main()
