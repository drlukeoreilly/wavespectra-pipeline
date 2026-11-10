# run_wavespectra_pipeline.py
# One-click pipeline that runs 01–04 with clean per-section output folders.
# Exports waterfalls, polar snapshots (all styles), NetCDF, PTM1_TRACK results,
# diagnostics, and a README that includes line numbers for user-editable settings.

from pathlib import Path
import re
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import textwrap

# Wavespectra must be installed in your 'waves' env
import wavespectra as ws  # registers .spec accessor

# ====================== CONFIG ======================
# [CFG] Base data folder (where Test.csv and wind csv live)
DATA_DIR = Path(r"C:\Users\LukeOReilly\OneDrive - GREEN REBEL GROUP\Desktop\IDS\Vattenfall\May 2025 Submission\03_QCed\03_DWR_FLS")
# [CFG] Filenames/patterns to locate spectra and wind files
SPECTRA_GLOB = ["Test.csv", "*waves_qced*.csv"]         # first match wins
WIND_GLOB    = ["u and v_speed_dir.csv", "*speed_dir*.csv"]
# [CFG] Local depth (m) for depth-aware partitioning
DEPTH_M      = 71.0
# [CFG] Datetime parsing (True means dd/mm/yyyy)
DAYFIRST     = True
# [CFG] Plot resolution
PNG_DPI      = 150

# Waterfall settings (01)
# [CFG] Max number of date ticks on x-axis
MAX_X_TICKS  = 8
# [CFG] Date label format
DATE_FMT     = "%Y-%m-%d"

# Directional snapshot (02)
# [CFG] Number of directional bins (e.g. 72 = 5° bins)
THETA_BINS   = 72
# [CFG] Which record to use if you want a fixed snapshot index (kept for reference)
SNAP_IDX     = 0

# NetCDF (03)
# [CFG] Compression level for netcdf4/h5netcdf
COMP_LEVEL   = 4

# PTM1_TRACK + diagnostics (04)
# [CFG] Number of swell time series to highlight alongside wind-sea
TOP_SWELLS       = 4
# [CFG] Wind speed threshold for "LOW_WIND" diagnostic
WIND_MIN_MS      = 4.0
# [CFG] Wind sector half-width for "DIR_MISMATCH" diagnostic
SECTOR_DEG       = 60.0
# [CFG] High-frequency floor (Hz) for "HF energy"
HF_MIN_HZ        = 0.15
# [CFG] Swell band ceiling (Hz)
SWELL_MAX_HZ     = 0.12
# [CFG] Swell dominance ratio (swell_energy / hf_energy)
SWELL_DOM_RATIO  = 3.0
# [CFG] If max available frequency < this → LIMITED_HF
HF_COVERAGE_MIN  = 0.35
# [CFG] Export PTM1 snapshots every N records (0 = off)
PTM1_SNAPSHOT_EVERY_N = 0
# [CFG] Export one PTM1 snapshot per day (True/False)
PTM1_SNAPSHOT_EVERY_DAY = True
# ====================================================


# ----------------- Common helpers -------------------
def format_date_axis(ax, fmt=DATE_FMT, maxticks=MAX_X_TICKS, rot=30):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=maxticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    plt.setp(ax.get_xticklabels(), rotation=rot, ha="right")
    plt.gcf().autofmt_xdate()

def find_first_existing(base: Path, patterns):
    for pat in patterns:
        for p in [*base.glob(pat)]:
            return p
    return None

def ensure_dir(p: Path) -> Path:
    p.mkdir(exist_ok=True)
    return p

def load_ef_from_csv(csv_path: Path, dayfirst=DAYFIRST) -> xr.DataArray:
    df = pd.read_csv(csv_path)
    time = pd.to_datetime(df.get("timestamp"), dayfirst=dayfirst, errors="coerce")

    pat = re.compile(r"^(\d+(?:\.\d+)?)Hz_SpectralDensity_M2pHz$")
    cols, f = [], []
    for c in df.columns:
        m = pat.match(str(c).strip())
        if m:
            cols.append(c); f.append(float(m.group(1)))

    if not cols:
        raise ValueError("No '<freq>Hz_SpectralDensity_M2pHz' columns found in spectra CSV.")

    order = np.argsort(f)
    f = np.array(f, dtype=float)[order]
    ef = df[cols].to_numpy(dtype=float)[:, order]

    da = xr.DataArray(
        ef, coords={"time": time, "freq": f}, dims=("time", "freq"),
        name="ef", attrs={"units": "m^2/Hz"}
    )
    return da, df, time, f

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

def build_efth(df, time, f, ef_cols, dir_cols, spr_cols, theta_bins=THETA_BINS):
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

        for i, _ in enumerate(f):
            efi = ef_row[i]
            if not (np.isfinite(efi) and efi > 0 and np.isfinite(th[i]) and np.isfinite(sp[i])):
                continue
            w = circ_gauss(thetas, th[i], sp[i])
            sw = w.sum()
            if sw > 0:
                w /= sw
            efth[it, i, :] = efi * w

    return xr.DataArray(
        efth, coords={"time": time, "freq": f, "dir": thetas},
        dims=("time", "freq", "dir"), name="efth", attrs={"units": "m^2/Hz/deg"}
    )

def interpolate_wind_to_wave_times(wind_df: pd.DataFrame, wave_time: pd.Series) -> pd.DataFrame:
    w = wind_df.copy()
    w["timestamp"] = pd.to_datetime(w["timestamp"], dayfirst=DAYFIRST, errors="coerce")
    w = w.sort_values("timestamp").dropna(subset=["timestamp"])
    spd  = pd.to_numeric(w["wind_speed"], errors="coerce")
    wdir = pd.to_numeric(w["wind_direction"], errors="coerce")
    # vector interpolation
    u = spd * np.sin(np.deg2rad(wdir))
    v = spd * np.cos(np.deg2rad(wdir))
    wind_vec = pd.DataFrame({"timestamp": w["timestamp"], "u": u, "v": v}).dropna()
    idx = pd.DataFrame({"timestamp": pd.to_datetime(wave_time)})
    mu = pd.merge_asof(
        idx.sort_values("timestamp"),
        wind_vec[["timestamp", "u"]].sort_values("timestamp"),
        on="timestamp", direction="nearest",
        tolerance=pd.Timedelta("2h"),  # was "2H"
    )
    mv = pd.merge_asof(
        idx.sort_values("timestamp"),
        wind_vec[["timestamp", "v"]].sort_values("timestamp"),
        on="timestamp", direction="nearest",
        tolerance=pd.Timedelta("2h"),  # was "2H"
    )
    uv = pd.DataFrame({"timestamp": idx["timestamp"], "u": mu["u"], "v": mv["v"]})
    uv["u"] = uv["u"].interpolate(limit_direction="both")
    uv["v"] = uv["v"].interpolate(limit_direction="both")
    speed = np.hypot(uv["u"], uv["v"])
    direction = (np.rad2deg(np.arctan2(uv["u"], uv["v"])) + 360) % 360
    return pd.DataFrame({"timestamp": idx["timestamp"], "wind_speed": speed, "wind_direction": direction})
# ----------------------------------------------------


# ================= SECTION 01 =======================
def section01_waterfall(csv_path: Path, out_root: Path):
    out_dir = ensure_dir(out_root / "01_ws_waterfall_ef")
    print(f"[01] Output → {out_dir}")

    da, _, _, _ = load_ef_from_csv(csv_path)

    # Peak frequency per time
    peak_f = np.full(da.sizes["time"], np.nan)
    for i in tqdm(range(da.sizes["time"]), desc="[01] peak freq"):
        y = da.isel(time=i).values
        if y.size and not (np.all(np.isnan(y)) or np.nanmax(np.abs(y)) == 0):
            peak_f[i] = da["freq"].values[np.nanargmax(y)]
    out_csv = out_dir / f"{csv_path.stem}_PeakFrequency.csv"
    pd.DataFrame({
        "time": pd.to_datetime(da.time.values),
        "f_peak_Hz": peak_f,
        "Tp_s": np.where(~np.isnan(peak_f), 1/peak_f, np.nan)
    }).to_csv(out_csv, index=False)
    print(f"[01] Saved: {out_csv}")

    # Waterfall plot
    valid = np.array([
        not (np.all(np.isnan(da.isel(time=i).values)) or np.nanmax(np.abs(da.isel(time=i).values)) == 0)
        for i in range(da.sizes["time"])
    ])
    if valid.any():
        sub = da.isel(time=np.where(valid)[0])
        fig, ax = plt.subplots(figsize=(12, 5))
        mesh = ax.pcolormesh(pd.to_datetime(sub["time"].values), sub["freq"].values, sub.T, shading="auto")
        ax.set_xlabel("Time"); ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"{csv_path.stem} — Time–Frequency Energy (ef)")
        cbar = fig.colorbar(mesh, ax=ax); cbar.set_label("m²/Hz")
        format_date_axis(ax)
        out_png = out_dir / f"{csv_path.stem}_waterfall_ef.png"
        fig.tight_layout(); fig.savefig(out_png, dpi=PNG_DPI); plt.close(fig)
        print(f"[01] Saved: {out_png}")
    else:
        print("[01] No usable spectra to plot.")


# ================= SECTION 02 =======================
def _fix_theta_labels(ax):
    """Force theta ticks at 0,45,...,315° with clean labels."""
    ticks_deg = np.arange(0, 360, 45)
    ax.set_xticks(np.deg2rad(ticks_deg))
    ax.set_xticklabels([f"{d}°" for d in ticks_deg])

def section02_directional_snapshot(csv_path: Path, df, time, f, out_root: Path):
    import numpy as np
    try:
        import cmocean  # noqa
        cm_thermal = cmocean.cm.thermal
    except Exception:
        cm_thermal = "RdBu_r"  # fallback if cmocean not installed

    out_dir = ensure_dir(out_root / "02_ws_directional_snapshot")
    print(f"[02] Output → {out_dir}")

    # Build efth
    f2, ef_cols, dir_cols, spr_cols = load_perfreq_columns(df)
    da2d = build_efth(df, time, f2, ef_cols, dir_cols, spr_cols, theta_bins=THETA_BINS)
    ds = xr.Dataset({"efth": da2d})

    # Choose first valid time
    idx = None
    for i in range(ds.sizes["time"]):
        arr = ds["efth"].isel(time=i).values
        if np.isfinite(arr).any() and np.nanmax(arr) > 0:
            idx = i; break
    if idx is None:
        print("[02] No valid efth snapshot to plot.")
        return ds

    snap = ds.isel(time=idx, drop=True)
    ts = pd.to_datetime(time.iloc[idx]).strftime("%Y%m%d_%H%M")

    # Helper to save and fix labels for single axis or facets
    def _save_fig(obj, path):
        try:
            axes = []
            if hasattr(obj, "axs"):      # FacetGrid
                axes = np.ravel(obj.axs)
            elif hasattr(obj, "ax"):     # xarray plot result
                axes = [obj.ax]
            else:
                axes = [plt.gca()]
            for ax in axes:
                _fix_theta_labels(ax)
            plt.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
            plt.close()
            print(f"[02] Saved: {path}")
        except Exception as e:
            print(f"[02] Failed saving {path.name}: {e}")

    # ========== Styles from the docs ==========
    # 0) Simplest usage (defaults: normalised, log radii)
    g = snap.spec.plot()
    _save_fig(g, out_dir / f"{csv_path.stem}_Efth_simple_{ts}.png")

    # 1) contour
    g = snap.spec.plot(kind="contour", colors="#af1607", linewidths=0.5)
    _save_fig(g, out_dir / f"{csv_path.stem}_Efth_contour_{ts}.png")

    # 2) contourf
    g = snap.spec.plot(kind="contourf", cmap=cm_thermal)
    _save_fig(g, out_dir / f"{csv_path.stem}_Efth_contourf_{ts}.png")

    # 3) pcolormesh
    g = snap.spec.plot(kind="pcolormesh", cmap=cm_thermal)
    _save_fig(g, out_dir / f"{csv_path.stem}_Efth_pcolormesh_{ts}.png")

    # 4) period radii (normalised, default)
    g = snap.spec.plot(as_period=True, cmap="pink_r")
    _save_fig(g, out_dir / f"{csv_path.stem}_Efth_period_norm_{ts}.png")

    # 5) period radii, real values (not normalised)
    g = snap.spec.plot(as_period=True, normalised=False, cmap="Spectral_r")
    _save_fig(g, out_dir / f"{csv_path.stem}_Efth_period_real_{ts}.png")

    # 6) manual log levels + cbar ticks
    levels = np.logspace(np.log10(0.005), np.log10(0.4), 15)
    g = snap.spec.plot(
        as_period=True, normalised=False, cmap="Spectral_r",
        levels=levels, cbar_ticks=[0.01, 0.1, 1]
    )
    _save_fig(g, out_dir / f"{csv_path.stem}_Efth_period_real_loglevels_{ts}.png")

    # 7) linear radii + custom ticks
    g = snap.spec.plot(
        as_period=True, normalised=False, levels=15, cmap="bone_r",
        logradius=False, radii_ticks=[5, 10, 15, 20, 25]
    )
    _save_fig(g, out_dir / f"{csv_path.stem}_Efth_period_linear_radii_{ts}.png")

    # 8) replicate deprecated as_log10 example
    ds1 = snap.where(snap > 0, 1e-5)
    ds1 = np.log10(ds1)
    g = ds1.spec.plot(
        as_period=True, logradius=False,
        cbar_kwargs={"label": "Normalised $\\log{E_{d}(f,d)}$"},
        vmin=0.39, levels=15, extend="both",
        cmap=cm_thermal,
    )
    _save_fig(g, out_dir / f"{csv_path.stem}_Efth_log10_style_{ts}.png")

    # 9) radii extents + custom labels
    g = snap.spec.plot(
        rmin=0, rmax=0.15, logradius=False, normalised=False, levels=25,
        cmap="gray_r", radii_ticks=[0.03, 0.06, 0.09, 0.12, 0.15],
        radii_labels=["0.05", "0.1", "0.15Hz"],
        radii_labels_angle=120, radii_labels_size=7,
    )
    _save_fig(g, out_dir / f"{csv_path.stem}_Efth_radii_extents_{ts}.png")

    # 10) xarray-parameter example (no colorbar, Turbo colormap)
    g = snap.spec.plot(kind="contourf", cmap="turbo", add_colorbar=False, extend="both", levels=25)
    _save_fig(g, out_dir / f"{csv_path.stem}_Efth_xarray_params_{ts}.png")

    return ds


# ================= SECTION 03 =======================
def pick_netcdf_engine():
    try:
        import netCDF4  # noqa
        return "netcdf4"
    except Exception:
        pass
    try:
        import h5netcdf  # noqa
        return "h5netcdf"
    except Exception:
        pass
    return "scipy"

def section03_export_netcdf(ds: xr.Dataset, csv_path: Path, out_root: Path):
    out_dir = ensure_dir(out_root / "03_ws_export_netcdf")
    print(f"[03] Output → {out_dir}")

    engine = pick_netcdf_engine()
    out_nc = out_dir / f"{csv_path.stem}.nc"

    # CF-compliant time encoding (prevents netcdf4 datetime64[ns] error)
    time_units = "seconds since 1970-01-01 00:00:00"
    time_calendar = "standard"

    if engine == "scipy":
        # netcdf3/scipy: no compression; still encode time as numeric with units
        encoding = {
            "efth": {"dtype": "float32"},
            "time": {"dtype": "int64", "units": time_units, "calendar": time_calendar},
            "freq": {"dtype": "float32"},
            "dir": {"dtype": "float32"},
        }
        print("[03] Writing with scipy (no compression).")
        ds.to_netcdf(out_nc, engine=engine, encoding=encoding)
    else:
        chunks_t = min(int(ds.sizes.get("time", 1)), 256)
        encoding = {
            "efth": {
                "zlib": True, "complevel": COMP_LEVEL, "dtype": "float32",
                "chunksizes": (chunks_t, ds.sizes["freq"], ds.sizes["dir"])
            },
            "time": {"dtype": "int64", "units": time_units, "calendar": time_calendar},
            "freq": {"dtype": "float32"},
            "dir": {"dtype": "float32"},
        }
        print(f"[03] Writing with {engine} (compressed).")
        ds.to_netcdf(out_nc, engine=engine, encoding=encoding)

    print(f"[03] Saved: {out_nc}")


# ================= SECTION 04 =======================
def ptm1_no_windsea_diagnostics(dset: xr.Dataset, dspart: xr.Dataset, wspd: np.ndarray, wdir: np.ndarray):
    times = pd.to_datetime(dset["time"].values)
    f = dset["freq"].values
    th = dset["dir"].values
    max_f = float(np.nanmax(f)) if f.size else np.nan

    stats = dspart.spec.stats(["hs"]).to_dataframe().reset_index()
    hs_ws = (stats[stats["part"] == 0].set_index("time")["hs"]).reindex(times, fill_value=np.nan)
    no_ws_mask = ~np.isfinite(hs_ws) | (hs_ws <= 0)

    rows = []
    for i, t in enumerate(times):
        if not no_ws_mask.iloc[i]:
            continue
        reasons = []
        if not np.isfinite(wspd[i]) or wspd[i] < WIND_MIN_MS:
            reasons.append("LOW_WIND")
        if not np.isfinite(max_f) or max_f < HF_COVERAGE_MIN:
            reasons.append("LIMITED_HF")

        if "efth" in dset:
            E = dset["efth"].isel(time=i).values
            if np.isfinite(E).any() and np.nanmax(E) > 0:
                wd = float(wdir[i]) if np.isfinite(wdir[i]) else np.nan
                if np.isfinite(wd):
                    d = np.minimum((th - wd) % 360, (wd - th) % 360)
                    sector_mask = d <= SECTOR_DEG
                else:
                    sector_mask = np.zeros_like(th, dtype=bool)

                hf_mask = f >= HF_MIN_HZ
                swell_mask = f <= SWELL_MAX_HZ
                hf_total = np.nansum(E[hf_mask, :])
                hf_wind  = np.nansum(E[hf_mask, :][:, sector_mask]) if sector_mask.any() else 0.0
                swell_en = np.nansum(E[swell_mask, :])

                if hf_total > 0 and (hf_wind / hf_total) < 0.15:
                    reasons.append("DIR_MISMATCH")
                if hf_total > 0 and (swell_en / hf_total) >= SWELL_DOM_RATIO:
                    reasons.append("SWELL_DOM")

        rows.append({
            "time": t,
            "wind_speed": float(wspd[i]) if np.isfinite(wspd[i]) else np.nan,
            "wind_direction": float(wdir[i]) if np.isfinite(wdir[i]) else np.nan,
            "hs_windsea": float(hs_ws.iloc[i]) if np.isfinite(hs_ws.iloc[i]) else 0.0,
            "reasons": ",".join(sorted(set(reasons))) if reasons else "UNSPECIFIED"
        })
    return pd.DataFrame(rows)

def section04_ptm1_track(ds: xr.Dataset, csv_path: Path, wind_df: pd.DataFrame, out_root: Path,
                         snapshot_every_n: int = PTM1_SNAPSHOT_EVERY_N,
                         snapshot_every_day: bool = PTM1_SNAPSHOT_EVERY_DAY):
    out_dir = ensure_dir(out_root / "04_ws_partition_demo")
    print(f"[04] Output → {out_dir}")

    time = pd.to_datetime(ds["time"].values)

    # Align wind
    wind_interp = interpolate_wind_to_wave_times(wind_df, time)
    ds = ds.copy()
    ds["wspd"] = (("time",), wind_interp["wind_speed"].to_numpy())
    ds["wdir"] = (("time",), wind_interp["wind_direction"].to_numpy())
    ds["dpt"]  = (("time",), np.full(len(time), DEPTH_M))

    stem = csv_path.stem

    # PTM1_TRACK
    dspart = ds.spec.partition.ptm1_track(wspd=ds.wspd, wdir=ds.wdir, dpt=ds.dpt)

    # Save stats CSV
    csv_stats = out_dir / f"{stem}_TRACKED_PTM1_timeseries.csv"
    dspart.spec.stats(["hs", "tp", "dpm"]).to_dataframe().reset_index().to_csv(csv_stats, index=False)
    print(f"[04] Saved: {csv_stats}")

    # Hs time series (wind-sea + top swells)
    df_stats = dspart.spec.stats(["hs", "tp", "dpm"]).to_dataframe().reset_index()
    df_stats = df_stats[df_stats["hs"] > 0]
    ws_df = df_stats[df_stats["part"] == 0]
    swell_df = df_stats[df_stats["part"] >= 1]
    swell_energy = swell_df.groupby("part")["hs"].apply(lambda x: (x**2).sum()).sort_values(ascending=False)
    top_sw = swell_energy.head(TOP_SWELLS).index.tolist()
    df_sel = pd.concat([ws_df, swell_df[swell_df["part"].isin(top_sw)]])
    others = swell_df[~swell_df["part"].isin(top_sw)].copy()
    if not others.empty:
        others["part"] = -1
        df_sel = pd.concat([df_sel, others])

    plt.figure(figsize=(11, 5))
    for pid, grp in df_sel.groupby("part"):
        label = "Wind-sea" if pid == 0 else ("Others" if pid == -1 else f"Swell {pid}")
        plt.plot(grp["time"], grp["hs"], marker="o", ms=2, lw=1, label=label)
    plt.ylabel("Hs (m)"); plt.xlabel("Time")
    plt.title("PTM1_TRACK — Wind-sea + Top Swells (Hs)")
    plt.grid(True, alpha=0.3); plt.legend(frameon=False)
    format_date_axis(plt.gca()); plt.tight_layout()
    out_png = out_dir / f"{stem}_PTM1_TRACK_Hs.png"
    plt.savefig(out_png, dpi=PNG_DPI); plt.close()
    print(f"[04] Saved: {out_png}")

    # --- Polar snapshots ---
    def _fix_theta_labels_local(ax):
        ticks_deg = np.arange(0, 360, 45)
        ax.set_xticks(np.deg2rad(ticks_deg))
        ax.set_xticklabels([f"{d}°" for d in ticks_deg])

    def _save_snapshot_at(i: int):
        snap = dspart.isel(time=i, drop=True)
        # choose best column dim
        col_dim = "part"
        for cand in ["part", "npart_id", "part_id"]:
            if cand in snap.dims:
                col_dim = cand; break
        g = snap.spec.plot(col=col_dim)
        import numpy as _np
        for ax in _np.ravel(g.axs):
            _fix_theta_labels_local(ax)
        ts = pd.to_datetime(time[i]).strftime("%Y%m%d_%H%M")
        out_pol = out_dir / f"{stem}_PTM1_TRACK_partition_snapshot_{ts}.png"
        plt.savefig(out_pol, dpi=PNG_DPI, bbox_inches="tight"); plt.close()
        print(f"[04] Saved: {out_pol}")

    # Single first-valid snapshot
    ok_idx = None
    for i in range(dspart.sizes["time"]):
        if "efth" in dspart:
            arr = dspart["efth"].isel(time=i).values
            if np.isfinite(arr).any() and np.nanmax(arr) > 0:
                ok_idx = i; break
    if ok_idx is not None:
        _save_snapshot_at(ok_idx)

    # Full-range exports (optional)
    if snapshot_every_n and snapshot_every_n > 0:
        for i in range(0, dspart.sizes["time"], snapshot_every_n):
            _save_snapshot_at(i)

    if snapshot_every_day:
    # first record position of each day (robust to pandas version)
        tnorm = pd.to_datetime(time).normalize().to_numpy()
        _, first_pos = np.unique(tnorm, return_index=True)
        for pos in np.sort(first_pos):
            if 0 <= int(pos) < dspart.sizes["time"]:
                _save_snapshot_at(int(pos))

    # Diagnostics + README
    wspd_al = ds["wspd"].values; wdir_al = ds["wdir"].values
    diag_df = ptm1_no_windsea_diagnostics(ds, dspart, wspd_al, wdir_al)
    if not diag_df.empty:
        out_diag = out_dir / f"{stem}_PTM1_no_windsea_diagnostics.csv"
        diag_df.to_csv(out_diag, index=False)
        print(f"[04] Saved: {out_diag}")

    # README with edit pointers (pull [CFG] lines and line numbers from this file)
    try:
        source = Path(__file__).read_text(encoding="utf-8").splitlines()
        cfg_lines = [(i+1, line.strip()) for i, line in enumerate(source) if "# [CFG]" in line]
    except Exception:
        cfg_lines = []

    try:
        hs_series = ds.spec.hs().to_series()
        hs_mean = float(np.nanmean(hs_series.values))
        hs_max  = float(np.nanmax(hs_series.values))
    except Exception:
        hs_mean = np.nan; hs_max = np.nan

    tmin, tmax = pd.to_datetime(ds["time"].values).min(), pd.to_datetime(ds["time"].values).max()
    n_times = int(ds.sizes["time"])
    wspd = wind_interp["wind_speed"].astype(float)
    wspd_stats = dict(mean=float(wspd.mean()), p90=float(wspd.quantile(0.9)))

    pct_no_ws = 0.0; tallies = {}; examples = []
    if not diag_df.empty:
        pct_no_ws = 100.0 * len(diag_df) / n_times
        tallies = diag_df["reasons"].value_counts().to_dict()
        examples = diag_df.head(10).to_dict(orient="records")

    edit_hints = "\n".join([f"  Line {ln:>4}: {txt}" for ln, txt in cfg_lines]) if cfg_lines else "  (Could not read source file for annotations.)"

    readme = f"""
    Wavespectra Pipeline — PTM1_TRACK Summary
    =========================================

    Inputs
    ------
    Spectra file : {csv_path.name}
    Wind file    : {find_first_existing(DATA_DIR, WIND_GLOB).name if find_first_existing(DATA_DIR, WIND_GLOB) else 'N/A'}
    Depth (m)    : {DEPTH_M}
    Output dir   : {out_dir}

    Time range
    ----------
    Start: {tmin}
    End  : {tmax}
    Records: {n_times}

    Wave overview
    -------------
    Mean Hs (m): {hs_mean:.2f}
    Max  Hs (m): {hs_max:.2f}

    Wind overview
    -------------
    Wind speed mean (m/s): {wspd_stats['mean']:.2f}
    Wind speed P90  (m/s): {wspd_stats['p90']:.2f}

    Wind-sea detection (PTM1_TRACK)
    --------------------------------
    Percent timestamps with NO wind-sea: {pct_no_ws:.1f}%
    Reason tallies:
    {tallies}

    Heuristics used
    ---------------
    - LOW_WIND: wspd < {WIND_MIN_MS} m/s
    - DIR_MISMATCH: <15% of high-frequency energy (f >= {HF_MIN_HZ} Hz) within ±{SECTOR_DEG}° of wind
    - SWELL_DOM: swell-band energy (f <= {SWELL_MAX_HZ} Hz) >= {SWELL_DOM_RATIO} × high-frequency energy
    - LIMITED_HF: max available freq < {HF_COVERAGE_MIN} Hz

    Where to edit settings (line numbers)
    -------------------------------------
    The lines below show where you can change inputs/thresholds/styles:
    {edit_hints}

    Sample 'no wind-sea' rows (first 10)
    ------------------------------------
    {pd.DataFrame(examples)}

    Files written in this section
    -----------------------------
    - {csv_stats.name}
    - {stem}_PTM1_TRACK_Hs.png
    - {stem}_PTM1_TRACK_partition_snapshot_*.png
    - {stem}_PTM1_no_windsea_diagnostics.csv (if any)
    """
    readme_path = out_dir / "README.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(readme))
    print(f"[04] Wrote README: {readme_path}")
# ----------------------------------------------------


# ======================== MAIN ======================
def main():
    # Locate files
    spec_path = find_first_existing(DATA_DIR, SPECTRA_GLOB)
    wind_path = find_first_existing(DATA_DIR, WIND_GLOB)
    if spec_path is None:
        raise FileNotFoundError(f"No spectra CSV found in {DATA_DIR} matching {SPECTRA_GLOB}")
    if wind_path is None:
        raise FileNotFoundError(f"No wind CSV found in {DATA_DIR} matching {WIND_GLOB}")

    print("Spectra file:", spec_path)
    print("Wind file   :", wind_path)

    # 01: Waterfall + peak frequency
    section01_waterfall(spec_path, DATA_DIR)

    # Load ef and dataframe once for reuse
    da, df, time, f = load_ef_from_csv(spec_path)

    # 02: Directional snapshot & return efth dataset for reuse
    ds = section02_directional_snapshot(spec_path, df, time, f, DATA_DIR)
    if ds is None:
        f2, ef_cols, dir_cols, spr_cols = load_perfreq_columns(df)
        da2d = build_efth(df, time, f2, ef_cols, dir_cols, spr_cols, theta_bins=THETA_BINS)
        ds = xr.Dataset({"efth": da2d})

    # 03: Export to NetCDF (CF-compliant time encoding)
    section03_export_netcdf(ds, spec_path, DATA_DIR)

    # 04: PTM1_TRACK + plots + diagnostics + README
    wind_df = pd.read_csv(wind_path)
    section04_ptm1_track(ds, spec_path, wind_df, DATA_DIR)

    print("\nAll sections complete ✅")

if __name__ == "__main__":
    main()
