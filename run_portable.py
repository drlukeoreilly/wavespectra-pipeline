"""Portable command-line entrypoint for the unchanged original pipeline.

Overrides only input/output configuration globals before invoking main().
No wave spectra or partitioning calculations are altered.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the archived wave-spectra pipeline with chosen input files")
    parser.add_argument("--spectra", type=Path, required=True, help="Spectra CSV (the per-frequency format documented in README)")
    parser.add_argument("--wind", type=Path, required=True, help="Wind CSV containing timestamp, wind_speed, wind_direction")
    parser.add_argument("--depth", type=float, default=71.0, help="Depth in metres (default matches archived script)")
    args = parser.parse_args()
    spectra = args.spectra.expanduser().resolve()
    wind = args.wind.expanduser().resolve()
    for p in (spectra, wind):
        if not p.is_file():
            parser.error(f"Input file does not exist: {p}")
    if spectra.parent != wind.parent:
        parser.error("The archived pipeline expects the spectra and wind files in the same folder. Place both inputs together.")
    if spectra == wind:
        parser.error("The spectra and wind files must be different")
    if args.depth <= 0:
        parser.error("Depth must be greater than zero")

    source = Path(__file__).resolve().parent / "original" / "run_wavespectra_pipeline.py"
    spec = importlib.util.spec_from_file_location("archived_wave_spectra_pipeline", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the original pipeline")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DATA_DIR = spectra.parent
    module.SPECTRA_GLOB = [spectra.name]
    module.WIND_GLOB = [wind.name]
    module.DEPTH_M = args.depth
    module.main()


if __name__ == "__main__":
    main()
