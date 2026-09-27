# Business Entity Resolution: reproducible pipeline

This pipeline regenerates `output/candidate_pairs.tsv` and `output/matching_results.tsv` from the challenge TSVs. Everything runs locally. There is no external lookup, API or data. The only pretrained weights referenced (optional cross-encoder, not used in the final submission) are MIT-licensed.

## Setup

Run these commands from the `student_resource/` directory (the one containing `dataset/` and `utils/`):

```bash
uv venv --python 3.11 .venv          # or: python3.11 -m venv .venv
uv pip install --python .venv/bin/python -r code/business_entity_resolution/requirements.txt
brew install libomp                  # macOS only (LightGBM runtime)
```

Expected data layout:

- `dataset/train/train_source{1,2,3}.tsv`
- `dataset/train/train_ground_truth.tsv`
- `dataset/test/test_source{1,2,3}.tsv`

## One command

```bash
bash code/business_entity_resolution/run_all.sh
```

Steps, each cached under `cache/` and keyed by its configuration (so reruns skip finished work):

| Step | Script | What it does |
|---|---|---|
| 1 | `scripts/run_blockers.py train ...` | Normalization (v3) plus 5 country-scoped TF-IDF blockers over all train S1 |
| 2 | `scripts/train_matcher.py --tag v2 --feat-tag v1 --top-n 12 --train-s1 600000 --final` | Union, then pruner (top-12), pair features, 5-fold stage-1 LightGBM, context and sibling features, stage-2 LightGBM, decision-rule sweep on fold 0, final stage-2 on all folds |
| 3 | `scripts/run_blockers.py test ...` | Same blockers on the test split |
| 4 | `scripts/predict_test.py --name v2_top12` | Test candidates, then `output/candidate_pairs.tsv`, then scoring and the decision rule, then `output/matching_results.tsv` |
| 5 | `utils/validate_submission.py --check-ids` | Official validator |

Resources:

- **Runtime:** about 9–10 h on an Apple M5 (10 cores, 16 GB RAM).
- **Peak memory:** about 7 GB.
- **Disk:** about 60 GB of cache.

## Source layout (`src/ber/`)

| Module | Purpose |
|---|---|
| `io.py` | TSV loading (tab-separated, no quoting) and parquet cache |
| `normalize.py` | Deterministic multi-representation normalization. Names: canonical legal forms, core name, sorted tokens, phonetic/transliteration skeleton, domain word-segmentation. Addresses: abbreviations (generic + country-scoped), region tokens, numbers, postal-like tokens |
| `splits.py` | 5-fold split grouped by S1 entity; fold 0 is the fixed validation fold |
| `metrics.py` | Exact challenge metric (macro F0.5 per S1, singleton rule) and candidate statistics |
| `blocking/tfidf.py` | Sparse TF-IDF top-k retrieval, forward (S1 → target) and reverse (target → S1), per country |
| `blocking/union.py` | Blocker registry, caching, union with retrieval features |
| `candidates.py` | Retrieval-feature pruner; the pruned set is exactly `candidate_pairs.tsv` |
| `features.py`, `pairs.py` | Pairwise string/numeric features (rapidfuzz, IDF overlaps, number/postal/region agreement), name frequency |
| `stage2.py`, `siblings.py` | Competition context (per-S1 and per-target, which encodes exclusivity) and sibling-consistency features |
| `decide.py` | Exclusivity and per-S1 expected-F0.5 subset selection |
| `errors.py` | Loss decomposition and FP/FN categorization |
| `experiment.py` | Experiment IDs, configs, metrics, `experiments/experiment_log.csv` |
| `models/crossencoder.py` | Optional multilingual-e5-small cross-encoder (not used in the final submission) |

`scripts/` also contains:

- screening scripts: `block_screen_*.py`, `screen_union.py`, `exp_stage2.py`
- `make_reports.py`, which writes `reports/*`

## Leakage safeguards

- Folds are assigned per S1 entity. Validation labels are never used for training in fold-0 evaluation.
- Stage-2 inputs derived from stage-1 are out-of-fold for every train pair.
- Blocking indexes and IDF tables use unlabelled text only.
- No test labels exist or are inferred.
