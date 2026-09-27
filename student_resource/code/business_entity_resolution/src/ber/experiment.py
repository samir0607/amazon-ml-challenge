"""Experiment bookkeeping: every run gets experiments/<id>/ with config + metrics and a
row appended to experiments/experiment_log.csv."""
import csv
import json
import platform
import resource
import time
from datetime import datetime
from pathlib import Path

from .paths import EXPERIMENTS

LOG = EXPERIMENTS / "experiment_log.csv"
LOG_FIELDS = ["exp_id", "timestamp", "stage", "description", "tier", "runtime_s", "peak_rss_gb",
              "pair_recall", "cand_mean", "cand_p95", "macro_f05", "micro_precision",
              "micro_recall", "f05_singletons", "decision", "notes"]


class Experiment:
    def __init__(self, name: str, stage: str, config: dict, description: str = "", tier: str = "screen"):
        self.id = f"{datetime.now():%Y%m%d_%H%M%S}_{name}"
        self.dir = EXPERIMENTS / self.id
        (self.dir / "logs").mkdir(parents=True, exist_ok=True)
        self.stage, self.config, self.description, self.tier = stage, config, description, tier
        self.metrics: dict = {}
        self.t0 = time.time()
        self.save_json("config.json", {**config, "_platform": platform.platform()})

    def save_json(self, name: str, obj) -> Path:
        p = self.dir / name
        p.write_text(json.dumps(obj, indent=2, default=str))
        return p

    def log(self, **metrics):
        self.metrics.update(metrics)

    def finish(self, decision: str = "", notes: str = "") -> dict:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
        self.metrics.update(runtime_s=round(time.time() - self.t0, 1), peak_rss_gb=round(rss, 2))
        self.save_json("metrics.json", self.metrics)
        new = not LOG.exists()
        with LOG.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            row = {"exp_id": self.id, "timestamp": f"{datetime.now():%Y-%m-%d %H:%M}", "stage": self.stage,
                   "description": self.description, "tier": self.tier, "decision": decision, "notes": notes}
            for k in LOG_FIELDS:
                if k in self.metrics:
                    v = self.metrics[k]
                    row[k] = round(v, 5) if isinstance(v, float) else v
            w.writerow(row)
        return self.metrics
