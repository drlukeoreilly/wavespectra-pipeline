"""Source-preservation/static tests; not a numerical validation suite."""
from pathlib import Path
import ast
import unittest

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "01_ws_waterfall_ef.py",
    "02_ws_directional_snapshot.py",
    "03_ws_export_to_netcdf.py",
    "04_ws_partition_demo.py",
    "run_wavespectra_pipeline.py",
]

class StaticTests(unittest.TestCase):
    def test_original_scripts_parse(self):
        for name in FILES:
            with self.subTest(name=name):
                ast.parse((ROOT / "original" / name).read_text(encoding="utf-8"))

    def test_entrypoint_exists(self):
        text = (ROOT / "run_portable.py").read_text(encoding="utf-8")
        self.assertIn("module.main()", text)
        self.assertIn('module.DATA_DIR = spectra.parent', text)

if __name__ == '__main__':
    unittest.main()
