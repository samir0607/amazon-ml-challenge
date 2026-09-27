"""Filesystem layout. Everything is resolved relative to the `student_resource/` root
(override with the BER_ROOT env var) so the official validator's relative paths work."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("BER_ROOT", Path(__file__).resolve().parents[4]))
DATA = ROOT / "dataset"
CACHE = Path(os.environ.get("BER_CACHE", ROOT / "cache"))
EXPERIMENTS = ROOT / "experiments"
REPORTS = ROOT / "reports"
OUTPUT = ROOT / "output"

for _d in (CACHE, EXPERIMENTS, REPORTS, OUTPUT):
    _d.mkdir(parents=True, exist_ok=True)
