# Initial audit — 10 November 2026

| Check | Finding | Action |
|---|---|---|
| Original source | Five Python scripts, unmodified, all compile | Preserve under `original/` |
| Consolidation | `run_wavespectra_pipeline.py` repeats logic of stand-alone scripts | Treat as authoritative launcher candidate; compare discrepancies before deduplicating |
| Portability | Hard-coded Windows OneDrive path to former employer/client in all scripts | `run_portable.py` overrides configuration; remove sensitive paths only with a separate reviewed change |
| Required packages | numpy, pandas, xarray, matplotlib, tqdm, wavespectra | Requirements documented; versions still to pin |
| NetCDF backend | Consolidated script picks netCDF4, h5netcdf or scipy; stand-alone export specifies compression | Install netCDF4 or h5netcdf for compressed export; test fallback |
| Scientific representation | E(f,theta) constructed from E(f), mean direction, spread with wrapped Gaussian-like kernel | This is reconstruction, not a measured 2D spectrum |
| Units/normalisation | Directional weights sum to 1 over bins, while output labelled m²/Hz/deg | Review angular bin-width division and energy-conservation tests |
| Direction conventions | Incoming wave *DirectionFrom* and wind directions enter PTM1_TRACK | Confirm conventions expected by Wavespectra and source wind CSV |
| Spectral matching | Some helper functions select frequencies by column-order and assume aligned energy/direction maps | Add shuffled-frequency and missing-column tests |
| Missing metadata | Timestamp timezone unspecified; fixed DAYFIRST=True | Explicit timezone/cadence checks |
| Missing directional fields | `row[dir_cols.get(..., np.nan)]` may fail if per-frequency column absent even where global fallback exists | Reproduce with missing-band inputs; fix only after baseline capture |
| Wind matching | Nearest-neighbour within two hours, then interpolated components | Test handling of unsorted/gapped records and original chronology |
| Numerics | Peak period computed as 1/fpeak, wind-sea partitions from wavespectra PTM1 | Validate with controlled synthetic spectrum and approved reference |
| Error handling | Stand-alone partition script catches failures and can continue without results; pipeline raises in main section | Add clearer failure reporting |
| Software rights | Original paths reference Green Rebel and Vattenfall | Keep private until ownership and client restrictions reviewed |

## Test status

**Passed:** syntax compilation of five untouched original sources; launcher argument help.

**Not yet tested:** end-to-end processing, NetCDF file writing, package compatibility, real-data comparison, numerical output validation.

## Next engineering steps

1. Confirm rights and obtain approved synthetic/redacted sample CSVs.
2. Record Windows Python version and pip freeze from working environment.
3. Test wave-column ordering, missing directional values, zero/negative frequencies, NaNs, timestamps and wind convention.
4. Verify angular energy conservation via numerical integration and ensure output units align with implementation.
5. Revisit angular reconstruction methods; make methodology explicit and compare to a measured directional spectrum where available.
6. Only then refactor repeated script routines and introduce CI tests.
