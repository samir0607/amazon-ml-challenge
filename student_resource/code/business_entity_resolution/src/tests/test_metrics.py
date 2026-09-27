"""Metric sanity checks against the README worked example and singleton rules.
Run (from code/business_entity_resolution): PYTHONPATH=src python src/tests/test_metrics.py"""
import polars as pl

from ber.metrics import macro_f05


def _df(pairs):
    return pl.DataFrame(pairs, schema=["s1_id", "match_id"], orient="row")


def test_readme_example_and_singletons():
    gt = _df([("S1-1", "S2-47"), ("S1-1", "S3-812"), ("S1-2", None), ("S1-3", None), ("S1-4", "S3-4")])
    pred = _df([("S1-1", "S2-47"), ("S1-1", "S2-193"), ("S1-1", "S3-812"), ("S1-3", "S2-9")])
    m = macro_f05(pred, gt, pl.Series(["S1-1", "S1-2", "S1-3", "S1-4"]))
    # S1-1: 0.714 (README), S1-2: correct empty -> 1, S1-3: false match on singleton -> 0,
    # S1-4: missed -> 0
    expected = (1.25 * (2 / 3) * 1.0 / (0.25 * (2 / 3) + 1.0) + 1 + 0 + 0) / 4
    assert abs(m["macro_f05"] - expected) < 1e-9, m
    assert m["singleton_false_match_rate"] == 0.5


if __name__ == "__main__":
    test_readme_example_and_singletons()
    print("ok")
