# Wave Spectra Pipeline — source archive and migration starter

**Status:** Historical source snapshot, not yet scientifically validated or end-to-end run in this environment. **Keep this repository private initially.**

## Overview

Processes tabular, per-frequency wave energy, direction and spreading inputs to produce:
- frequency-time waterfall plots and peak period CSVs;
- illustrative reconstructed directional-spectrum figures;
- multidimensional NetCDF exports;
- Wavespectra `PTM1_TRACK` wind-sea/swell partitions and diagnostics.

The `original/` directory contains **five untouched scripts** uploaded 10 November 2026. `original/run_wavespectra_pipeline.py` is the consolidated pipeline and overlaps substantially with the four stand-alone scripts. The new `run_portable.py` is a launch adapter that supplies input paths/depth; it does not change numerical routines.

## Inputs

Place the spectra and wind CSVs in the **same folder** (required by original implementation). Spectra CSV requires `timestamp` and columns such as `0.100Hz_SpectralDensity_M2pHz`; directional processing uses matching `0.100Hz_DirectionFrom_deg` and `0.100Hz_Spread_deg` columns or global `MeanDir_deg` / `MeanSpread_deg`. Wind CSV requires `timestamp`, `wind_speed`, `wind_direction` (meteorological convention must be confirmed). Timestamps assumed day-first by default.

**Do not commit private client CSVs** or data-derived plots without clearance.

## Run (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_portable.py --spectra "C:\\path\\to\\waves.csv" --wind "C:\\path\\to\\wind.csv" --depth 71
```

The launcher writes outputs into subfolders of the source CSV directory (matching original behaviour). On a machine with Wavespectra installed, to check launch options without processing data:

```powershell
py run_portable.py --help
```

The legacy scripts in `original/` contain a hard-coded path to a former employer/client project; **do not run them directly without changing that setting or use the portable launcher instead**.

## Scientific limitations

The input CSV contains omnidirectional spectral energy and directional summary statistics. The Gaussian-like angular spreading model used here **reconstructs an illustrative E(f,θ)**, not the measured underlying directional spectrum. The weights are normalised by the sum over direction bins, and the existing `m²/Hz/deg` label should be reviewed for directional-bin width before interpreting or comparing absolute spectral densities. The existing analysis also assumes consistent wind and wave direction conventions, even though the input columns use *DirectionFrom*.

`PTM1_TRACK` results depend on the Wavespectra package version and on reliable wind input, depth and full angular spectrum. The wind assignment relies on nearest matching and interpolation across gaps; check alignment and units. Treat the 'no wind-sea' labels as **heuristics**, not validation of partition physical accuracy.

## Verification status

- All five original Python scripts passed Python syntax compilation.
- `run_portable.py` passed Python syntax compilation and help-argument smoke test.
- End-to-end numerical execution **not performed**: Wavespectra is not installed in the audit environment; no input CSVs were provided.
- Dependency versions are **not pinned** to a validated environment.

See `docs/MIGRATION_AUDIT.md` for follow-up checks.

## Source and ownership

These scripts contain a path mentioning a former employer and a named offshore-wind client. **Confirm code ownership and sharing rights before any public release.** No real data files are included.
