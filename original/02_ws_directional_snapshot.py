# 02_ws_directional_snapshot.py
from pathlib import Path
import re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIG ---
CSV_PATH = Path(r"C:\Users\LukeOReilly\OneDrive - GREEN REBEL GROUP\Desktop\IDS\Vattenfall\May 2025 Submission\03_QCed\03_DWR_FLS\Test.csv")
DAYFIRST = True
PNG_DPI = 150
THETA_BINS = 72  # 5° bins
# -------------

def load_columns(csv_path: Path):
    df = pd.read_csv(csv_path)
    time = pd.to_datetime(df.get("timestamp"), dayfirst=DAYFIRST, errors="coerce")

    pat_ef  = re.compile(r"^(\d+(?:\.\d+)?)Hz_SpectralDensity_M2pHz$")
    pat_dir = re.compile(r"^(\d+(?:\.\d+)?)Hz_DirectionFrom_deg$")
    pat_spr = re.compile(r"^(\d+(?:\.\d+)?)Hz_Spread_deg$")

    f_list, ef_cols = [], []
    dir_cols, spr_cols = {}, {}

    for c in df.columns:
        s = str(c).strip()
        m_ef = pat_ef.match(s)
        if m_ef:
            ef_cols.append(c)
            f_list.append(float(m_ef.group(1)))
        m_dir = pat_dir.match(s)
        if m_dir:
            dir_cols[float(m_dir.group(1))] = c
        m_spr = pat_spr.match(s)
        if m_spr:
            spr_cols[float(m_spr.group(1))] = c

    if not ef_cols:
        raise ValueError("No ef(f) columns found (expected '<freq>Hz_SpectralDensity_M2pHz').")

    f = np.sort(np.array(f_list))
    return df, time, f, ef_cols, dir_cols, spr_cols

def first_valid_row(df, ef_cols):
    for i in range(len(df)):
        y = df[ef_cols].iloc[i].to_numpy(dtype=float)
        if y.size and not (np.all(np.isnan(y)) or np.nanmax(np.abs(y)) == 0):
            return i
    return None

def main():
    tqdm.write(f"Reading {CSV_PATH}")
    df, time, f, ef_cols, dir_cols, spr_cols = load_columns(CSV_PATH)
    idx = first_valid_row(df, ef_cols)
    if idx is None:
        print("No usable spectrum found."); return

    # Map ef(f) values to sorted frequency vector
    f_from_col = lambda c: float(re.match(r"^(\d+(?:\.\d+)?)", c).group(1))
    ef_map = {f_from_col(c): df[c].iloc[idx] for c in ef_cols}
    ef = np.array([ef_map[fi] for fi in f], dtype=float)

    # Pull per-frequency direction/spread if present
    theta = np.array([df[dir_cols[fi]].iloc[idx] if fi in dir_cols else np.nan for fi in f], dtype=float)
    spread = np.array([df[spr_cols[fi]].iloc[idx] if fi in spr_cols else np.nan for fi in f], dtype=float)

    # Fallback to global MeanDir/MeanSpread if per-frequency missing
    if (not np.isfinite(theta).any()) and ("MeanDir_deg" in df.columns):
        theta[:] = float(df["MeanDir_deg"].iloc[idx])
    if (not np.isfinite(spread).any()) and ("MeanSpread_deg" in df.columns):
        spread[:] = max(float(df["MeanSpread_deg"].iloc[idx]), 5.0)

    ts = time.iloc[idx]

    # --- Plot 1: Direction vs Frequency (if theta available) ---
    if np.isfinite(theta).any():
        plt.figure()
        plt.plot(f, theta, marker="o", linewidth=1)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Mean Direction From (deg)")
        plt.title(f"{CSV_PATH.stem} — Direction vs Frequency\n{ts}")
        plt.grid(True, linestyle=":")
        out1 = CSV_PATH.with_name(CSV_PATH.stem + "_dir_vs_freq.png")
        plt.tight_layout(); plt.savefig(out1, dpi=PNG_DPI); plt.close()
        print("Saved:", out1)
    else:
        print("No direction data found; skipped direction-vs-frequency plot.")

    # --- Plot 2: Polar E(f,θ) snapshot using a simple circular Gaussian spread ---
    have_dir_spread = np.isfinite(theta).any() and np.isfinite(spread).any()
    if have_dir_spread and np.isfinite(ef).any() and np.nanmax(ef) > 0:
        thetas = np.linspace(0, 360, THETA_BINS, endpoint=False)

        def wrapped_normal_pdf(tdeg, mu, sig):
            # circular "Gaussian-like" around mu with spread sig (deg)
            d = np.minimum(np.mod(tdeg - mu, 360), np.mod(mu - tdeg, 360))
            return np.exp(-(d ** 2) / (2 * (max(sig, 1e-3) ** 2)))

        efth = np.zeros((len(f), len(thetas)), dtype=float)
        for i in range(len(f)):
            if not (np.isfinite(ef[i]) and ef[i] > 0 and np.isfinite(theta[i]) and np.isfinite(spread[i])):
                continue
            w = wrapped_normal_pdf(thetas, theta[i], spread[i])
            w = w / (w.sum() if w.sum() > 0 else 1.0)
            efth[i, :] = ef[i] * w

        # Polar heatmap
        th_rad = np.deg2rad(np.r_[thetas, thetas[0]])
        F, TH = np.meshgrid(f, th_rad)
        Z = np.vstack([efth.T, efth.T[:1, :]])  # close circle

        ax = plt.subplot(111, projection="polar")
        pcm = ax.pcolormesh(TH, F, Z, shading="auto")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(f"{CSV_PATH.stem} — Directional snapshot E(f,θ)\n{ts}")
        cbar = plt.colorbar(pcm, pad=0.1); cbar.set_label("m²/Hz/deg")
        out2 = CSV_PATH.with_name(CSV_PATH.stem + "_directional_snapshot.png")
        plt.savefig(out2, dpi=PNG_DPI, bbox_inches="tight"); plt.close()
        print("Saved:", out2)
    else:
        print("No usable per-frequency direction/spread; skipped directional snapshot.")

    print("Done ✅")

if __name__ == "__main__":
    main()
