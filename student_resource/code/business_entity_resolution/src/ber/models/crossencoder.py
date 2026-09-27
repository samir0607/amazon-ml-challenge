"""Pairwise cross-encoder (multilingual-e5-small, MIT, 118M params) scored only on
uncertain candidate pairs. Runs on Apple MPS (fp16 autocast) or CUDA if available.

Input: raw (un-normalized) name/address/country for both sides, explicit field labels
and a missing marker, so the model sees transliteration / script / format noise
directly.
"""
import math
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..paths import CACHE

MODEL_NAME = "intfloat/multilingual-e5-small"
HF_HOME = str(CACHE / "hf")
MISSING = "[none]"


def device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def side_text(name, addr, country):
    return f"name: {name or MISSING} | address: {addr or MISSING} | country: {country or MISSING}"


def pair_texts(pairs: pl.DataFrame, split: str) -> tuple[list, list]:
    """pairs: (q, t) -> (s1 texts, target texts) aligned with pairs row order."""
    from ..data import load_queries, load_targets
    cols = ["business_name", "business_address", "country"]
    q = load_queries(split, cols)
    t = load_targets(split, cols)
    j = (pairs.select("q", "t").with_row_index("_i")
         .join(q.rename({c: f"q_{c}" for c in cols}), on="q", how="left")
         .join(t.rename({c: f"t_{c}" for c in cols}), on="t", how="left").sort("_i"))
    a = [side_text(n, ad, c) for n, ad, c in zip(j["q_business_name"], j["q_business_address"], j["q_country"])]
    b = [side_text(n, ad, c) for n, ad, c in zip(j["t_business_name"], j["t_business_address"], j["t_country"])]
    return a, b


class CrossEncoder:
    def __init__(self, path: str | None = None, max_len: int = 96):
        self.dev = device()
        src = path or MODEL_NAME
        self.tok = AutoTokenizer.from_pretrained(src, cache_dir=HF_HOME)
        self.model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=1, cache_dir=HF_HOME).to(self.dev)
        self.max_len = max_len
        self.dtype = torch.float16 if self.dev in ("mps", "cuda") else torch.bfloat16

    def _enc(self, a, b):
        # fixed-length padding: constant tensor shapes stop MPS from caching a new graph per length
        e = self.tok(a, b, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        return {k: v.to(self.dev) for k, v in e.items()}

    def train(self, a, b, y, epochs=1, bs=64, lr=4e-5, warmup=0.06, log_every=50, save_to=None, seed=0):
        rng = np.random.default_rng(seed)
        n = len(y)
        steps = epochs * math.ceil(n / bs)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / max(1, warmup * steps)) * max(0.0, (steps - s) / steps))
        self.model.train()
        step, t0, run = 0, time.time(), 0.0
        yt = torch.tensor(np.asarray(y, dtype=np.float32))
        for _ in range(epochs):
            order = rng.permutation(n)
            for i in range(0, n, bs):
                idx = order[i:i + bs]
                enc = self._enc([a[j] for j in idx], [b[j] for j in idx])
                with torch.autocast(self.dev, dtype=self.dtype, enabled=self.dev != "cpu"):
                    logits = self.model(**enc).logits.squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), yt[idx].to(self.dev))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
                step += 1
                if step % log_every == 0:
                    print(f"  ce step {step}/{steps} loss {run:.4f} {step * bs / (time.time() - t0):.0f} pairs/s", flush=True)
                if save_to and step % 2000 == 0:
                    self.save(save_to)
        if save_to:
            self.save(save_to)

    def save(self, path):
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)

    @torch.no_grad()
    def predict(self, a, b, bs=256, log_every=50):
        self.model.eval()
        # length-sorted batching for speed; results restored to input order
        order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])
        out = np.empty(len(a), dtype=np.float32)
        t0 = time.time()
        for k, i in enumerate(range(0, len(a), bs)):
            idx = order[i:i + bs]
            enc = self._enc([a[j] for j in idx], [b[j] for j in idx])
            with torch.autocast(self.dev, dtype=self.dtype, enabled=self.dev != "cpu"):
                out[idx] = self.model(**enc).logits.squeeze(-1).float().cpu().numpy()
            if (k + 1) % log_every == 0:
                print(f"  ce predict {i + bs}/{len(a)} {(i + bs) / (time.time() - t0):.0f} pairs/s", flush=True)
        return out  # logits
