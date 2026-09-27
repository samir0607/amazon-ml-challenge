"""Serialize heavy jobs on the shared 16 GB machine: `heavy_lock.py -- <cmd ...>` waits
for an exclusive lock on cache/.heavy.lock, runs the command, then releases it."""
import fcntl
import subprocess
import sys
import time
from pathlib import Path

lock = Path(__file__).resolve().parents[4] / "cache" / ".heavy.lock"
cmd = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
with open(lock, "w") as f:
    t0 = time.time()
    fcntl.flock(f, fcntl.LOCK_EX)
    print(f"[heavy_lock] acquired after {time.time() - t0:.0f}s: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
sys.exit(rc)
