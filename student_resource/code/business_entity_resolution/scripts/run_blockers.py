"""Run a list of blockers over ALL S1 of a split (cached per blocker)."""
import sys
import time

from ber.blocking.union import run_blocker

SPECS = {
    "rev_nameaddr": {"name": "rev_nameaddr", "kind": "rev", "cfg": "nameaddr_word", "k": 5},
    "rev_addr": {"name": "rev_addr", "kind": "rev", "cfg": "addr_word", "k": 3},
    "rev_name": {"name": "rev_name", "kind": "rev", "cfg": "name_word", "k": 3},
    "fwd_nameaddr": {"name": "fwd_nameaddr", "kind": "fwd", "cfg": "nameaddr_word", "k": 30},
    "fwd_addr": {"name": "fwd_addr", "kind": "fwd", "cfg": "addr_word", "k": 10},
    "rev_skel_nonascii": {"name": "rev_skel_nonascii", "kind": "rev", "cfg": "skel_char", "k": 3, "subset": "nonascii_name"},
    "exact_name_sorted": {"name": "exact_name_sorted", "kind": "exact", "cfg": {"key": "name_sorted", "max_block": 50}},
    "exact_addr_sorted": {"name": "exact_addr_sorted", "kind": "exact", "cfg": {"key": "addr_sorted", "max_block": 50}},
}

if __name__ == "__main__":
    split = sys.argv[1]
    for n in sys.argv[2:]:
        t0 = time.time()
        c = run_blocker(split, SPECS[n])
        print(f"DONE {split} {n}: {c.height} pairs in {time.time() - t0:.0f}s", flush=True)
