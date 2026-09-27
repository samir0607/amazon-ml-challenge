# Current best pipeline

**Validation: macro F0.5 = 0.98218** on held-out half B of fold 0 (v4 = v3 cascade + cross-encoder fusion; base 0.97179 on the same half). The GBDT-only v3 scores 0.97217 on all of fold 0.
5-fold CV of the decision rule: 0.97101 ± 0.00004.
Candidate set: learned-blocking cascade, 4.65 candidates per S1 on validation (5.12 on test), pair recall 0.9797.

## Pipeline
1. **Normalization v3.** Multi-representation names: canonical legal forms, core name, sorted tokens, phonetic/transliteration skeleton, domain-name word segmentation. Addresses: abbreviation expansion (generic + France-scoped), region tokens, numbers, postal.
2. **Blocking.** Union of 5 TF-IDF word retrievers, all country-scoped:
   - reverse (target → S1) name+address, top-5
   - reverse address, top-3
   - reverse name, top-3
   - forward (S1 → target) name+address, top-30
   - forward address, top-10

   A LightGBM pruner on retrieval features keeps the top-12 per S1. That pruned set is `candidate_pairs.tsv`.
3. **Pair features.** 48 string/numeric features (rapidfuzz, IDF-weighted overlaps, numbers/postal/region agreement), name-frequency features, and retrieval features.
4. **Stage-1 LightGBM**, cross-fitted over 5 folds, gives an out-of-fold p1 for every pair.
5. **Context features from p1:**
   - per-S1 rank and gap
   - per-target competition: margin over the best other S1 (exclusivity)
   - set-level: expected match count, number of candidates above 0.5 / 0.2
6. **Sibling-consistency features.** Similarity of the target to the S1's other plausible candidates, weighted by p1.
7. **Stage-2 LightGBM** on all of the above.
8. **Decision rule.** Exclusivity (each target keeps only its best S1), then per-S1 expected-F0.5 subset selection with min_p = 0.6 and extra_mass = 0.2.

## Why this configuration
Each component was kept only on a measured fold-0 gain (see experiments/experiment_log.csv):

| Step | Change |
|---|---|
| reverse retrieval | top-1 recall 93.7% at 4.7 cands vs forward top-10 91.9% at 10 |
| v3 normalization | reverse top-1 recall 93.7% → 94.6% |
| stage-2 context vs stage-1 | ~+0.004 |
| tuned expected-F rule | +0.0003 (held-out half) |
| set-level features | +0.0012 |
| sibling features | +0.0018 |
| more stage-2 training data (300k → 900k S1) | +0.0011 (measured without siblings) |
| v2: stage-1 on 600k S1/fold, stage-2 on 1.2M S1 (easy negatives subsampled 25%, weighted) | 0.97021 → 0.97205 |

Decision rule: min_p = 0.6 was chosen over 0.5. Their F0.5 is tied (0.97205 vs 0.97207), but 0.6 does much better on singletons (0.969 vs 0.959). That protects against a different singleton rate on test, e.g. the unseen France data.
| v3: learned-blocking cascade (stage-1 p1 ≥ 0.01 defines `candidate_pairs.tsv`); stage 2 retrained on survivors | 0.97205 → 0.97217, 12 → 4.65 candidates per S1 |
| 5-fold CV of the decision rule and τ | final rule best on 4/4 folds |
| Rejected: non-Latin skeleton blocker | about +0.00005 |
| v4: cross-encoder (e5-small) on borderline pairs + logistic fusion | +0.0104 on held-out half B (0.97179 → 0.98218) |
| **Final: round-2 cross-encoder fusion + prior odds ×0.25** | public leaderboard **0.975** (full fusion ×1: 0.971; add-only: 0.950) |
