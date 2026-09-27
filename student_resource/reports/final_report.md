# Business Entity Resolution: Final Report

**Final validation score: macro F0.5 ≈ 0.982** with the cross-encoder fusion. This is **0.98218** on held-out half B of fold 0 (about 220k S1 never seen by the cross-encoder or the fusion model), versus 0.97179 without it on the same half. The GBDT-only pipeline scores 0.97217 on all of fold 0 (441,365 held-out S1, matched against the full train S2/S3 index).

The final candidate set is the output of a three-step blocking cascade. It averages **4.65 candidates per S1 on validation and 5.12 on test** (p95 = 9, max = 12), keeping 97.97% of true pairs. The decision rule and the cascade threshold were confirmed by 5-fold cross-validation (mean 0.97101, std 0.00004 across folds). The submission passes the official validator (`--check-ids`, no subset warnings).

---

## 1. Dataset summary

| Split | S1 | S2 | S3 | Countries |
|---|---|---|---|---|
| train | 2,206,821 | 5,034,616 | 5,285,603 | US, India |
| test | 1,732,544 | 4,887,273 | 5,082,316 | India, US, **France (15% of test S1, unseen in training)** |

Measured facts that drove the design (see `reports/dataset_report.md`):

- **Singletons.** 5.6% of train S1 entities have no match. The mean is 3.46 matches per S1, and the maximum is 11.
- **Exclusivity.** Every S2/S3 record matches at most one S1 (true for 100% of labelled records), and 26% of S2/S3 records match nothing.
- **Country.** The country label agrees on 100% of true pairs, so blocking is scoped by country. Country is treated as an open set of string labels.
- **Name noise:**
  - word shuffles and doubled words
  - character and leet typos (`Denta1`, `Si1ver`)
  - diacritics
  - legal-suffix swaps
  - prefixes like `--` and `Sri`
  - `X trading as Y` and `formerly:` constructions
  - domain-style names (`pediatricdentalcenter.com`)
  - randomly replaced names
  - Indic-script names: 15% of S2 names in train. Devanagari is most common, alongside Tamil, Kannada, Telugu, Malayalam, Bengali and Gujarati.
- **Address noise:**
  - uppercase and abbreviations
  - component reordering
  - state name vs state code
  - injected wrong city
  - native-script state names
  - partial or missing addresses (3.3% of S2/S3)
- **Generic names are common.** "Heritage Alliance" and "Cardiology Medicine Inc" each exist as many distinct businesses. Name-only retrieval tops out at 68% recall at K=50.
- **The test set is denser.** It has 5.75 targets per S1 versus 4.67 in train, so it likely contains more unmatched distractor records.

## 2. Validation protocol

- **Grouped split.** 5 folds by S1 entity, from a seeded permutation of sorted S1 IDs. No S1 appears in two folds, and no pair-level split is used anywhere.
- **Fold 0 is the fixed primary validation fold** (441,365 S1). It was never redesigned. Screening used a fixed subset of 50,000 S1 from fold 0, always queried against the **full** 10.3M-record target index so that candidate density is realistic.
- **Label-free indexes.** Blocking indexes and IDF tables are built from S2/S3 text only, with no labels, exactly as on test.
- **Stacking without leakage.** Stage-1 scores used by stage 2 are out-of-fold: every train pair is scored by a model that did not train on its S1. Stage 2 trains on folds 1–4 and is evaluated on fold 0.
- **Decision-rule tuning** used half of fold 0 and was reported on the other half.
- **Metric.** `ber/metrics.py` computes F0.5 per S1 and macro-averages it, with singletons scoring 1 for an empty prediction and 0 otherwise. It is unit-tested against the README worked example (0.714).

## 3. Blocking experiments

All rows are measured on the 50k screening S1 against the full index; the final row uses all 441k fold-0 S1. "Reverse" means every S2/S3 record queries the S1 index for its top-k S1; that S1 receives the record as a candidate.

| Experiment | Candidate Recall | Avg Candidates | P95 Candidates |
|---|---|---|---|
| Exact key: sorted core name | 0.4650 | 4.39 | 17 |
| Exact key: sorted address tokens | 0.3593 | 1.53 | 5 |
| Char-3gram TF-IDF, name only, top-50 | 0.6777 | 50 | 50 |
| Word TF-IDF name+skeleton+address, forward top-10 | 0.9185 | 10 | 10 |
| Word TF-IDF name+skeleton+address, forward top-50 | 0.9573 | 50 | 50 |
| **Reverse** name+address top-1 (norm v1) | 0.9373 | 4.67 | 8 |
| **Reverse** name+address top-1 (norm v3) | **0.9464** | **4.67** | 8 |
| Reverse name+address top-3 (norm v3) | 0.9648 | 13.95 | 36 |
| Union of all 7 blockers | 0.9856 | 60.56 | 129 |
| Union minus reverse name+address | 0.9767 | 50.50 | 89 |
| Union minus forward name+address | 0.9773 | 42.64 | 119 |
| Union minus exact name / exact address | 0.9855 / 0.9856 | 59.9 / 60.6 | 129 |
| Union pruned (retrieval-feature LightGBM) top-8 | 0.9753 | 8 | 8 |
| Union pruned top-10 | 0.9802 | 10 | 10 |
| **Union pruned top-12 (final)** | **0.9819** | **12** | **12** |
| Union pruned top-15 | 0.9834 | 15 | 15 |
| Union pruned top-20 | 0.9844 | 20 | 20 |
| Top-12 set on full fold 0 | 0.9816 | 12 | 12 |
| Cascade: top-12 then stage-1 p1 ≥ 0.01, full fold 0 | 0.9797 | 4.64 | 8 |
| **Cascade p1 ≥ 0.01, final (stage 2 retrained on survivors)** | **0.9797** | **4.65** | **8** |
| Cascade p1 ≥ 0.03 | 0.9763 | 4.03 | 7 |
| Cascade p1 ≥ 0.05 | 0.9746 | 3.88 | 7 |
| Pruner-score threshold instead (retrieval features only), comparable size | 0.9596 | 4.67 | 8 |

Decisions:

- **Reverse retrieval is the biggest structural win.** It exploits exclusivity: the correct S1 is usually a target's best-scoring S1.
- **The exact-key blockers were dropped.** They add no more than 0.0001 recall on top of TF-IDF.
- **Top-12 was chosen.** Going to top-15 would add only 0.0015 recall for 25% more pairs.
- **Char-level TF-IDF over all queries was rejected on cost.** It took 326 s per 50k queries, about 4 hours for the full split, for low recall.
- **The final candidate set is a learned-blocking cascade.** Smaller candidate sets rank higher in the final evaluation, so a third blocking step keeps only pairs whose stage-1 pair score (string-feature LightGBM, cross-fitted) reaches 0.01. Stage 2 then runs inference only over those survivors, which are exactly `candidate_pairs.tsv`. This cut candidates from 12 to 4.65 per S1 with no F0.5 loss: 0.97165 → 0.97173 at equal training size.
- **A retrieval-only threshold at a similar size loses 0.004 F0.5**, so the learned cascade is essential.
- **A 6th blocker for non-Latin names was rejected.** It is a reverse char-TF-IDF over phonetic skeletons of Indic-script targets. It adds 0.05% of true pairs, of which only about 40% are matchable, for about +0.00005 F0.5.

## 4. Matcher experiments

All rows are on fold 0 (441,365 S1) with the full candidate set.

| Model | Precision (micro) | Recall (micro) | Macro F0.5 |
|---|---|---|---|
| Stage-1 LightGBM (pair + retrieval features), expected-F rule | 0.9912 | 0.9240 | 0.9649 |
| Two-stage v1 (+ competition context), expected-F | 0.9909 | 0.9289 | 0.9669 |
| + tuned rule (min_p 0.6, extra mass 0.2) | — | — | 0.9672 |
| + set-level features (expected matches / #candidates above 0.5 per S1) | — | — | 0.9685 |
| + more stage-2 data (300k → 900k S1) | — | — | 0.9695 |
| + sibling-consistency features (300k S1) | — | — | 0.9702 |
| v2: stage-1 on 600k S1 per fold, stage-2 on 1.2M S1, set + sibling features (12 candidates) | 0.9926 | 0.9415 | 0.97205 |
| v3: v2 + cascade (p1 ≥ 0.01, 4.65 candidates), stage 2 retrained on survivors | 0.9926 | 0.9420 | 0.97217 |
| **v4 final: v3 + cross-encoder (multilingual-e5-small, MIT) on borderline pairs, logistic fusion; fold-0 half B** | — | — | **0.98218** (base 0.97179 on the same half) |

Decision rules for the v2 scores:

| Rule | Precision | Recall | Macro F0.5 | Singleton F0.5 |
|---|---|---|---|---|
| threshold 0.5 | 0.9853 | 0.9526 | 0.96966 | 0.9589 |
| threshold 0.7 | 0.9924 | 0.9424 | 0.97162 | 0.9767 |
| expected-F, no gate | 0.9925 | 0.9412 | 0.97199 | 0.9508 |
| **expected-F, min_p 0.6 (final)** | **0.9926** | **0.9415** | **0.97205** | **0.9689** |

## 5. Ablations

`reports/ablation_results.csv` and `reports/candidate_analysis.csv` list every run. Ablations actually measured:

| Component | Effect on fold 0 |
|---|---|
| Normalization v1 → v3 (domain segmentation, France abbreviations, number suffixes) | reverse top-1 recall +0.0091 |
| Reverse retrieval (leave-one-out from the union) | −0.0089 recall without it |
| Forward name+address (leave-one-out) | −0.0083 recall without it |
| Exact-key blockers | ≤ +0.0001 recall, dropped |
| Stage-2 context over stage-1 only | about +0.003 to +0.005 F0.5 |
| Set-level features | +0.0012 F0.5 |
| Sibling / cross-source consistency features | +0.0018 F0.5 |
| Stage-2 training data 300k → 900k S1 | +0.0011 F0.5 |
| Exclusivity post-pass | +0.00004 (stage 2 already learns it via competition features; kept as a guarantee) |
| Tuned expected-F gate | +0.0003 on the held-out half of fold 0 |
| Learned-blocking cascade p1 ≥ 0.01 (12 → 4.65 candidates per S1) | +0.00008 (0.97165 → 0.97173, same training size) |
| 5-fold CV of the decision rule (exclusivity scope × expected-F grid × thresholds × cascade τ) | final rule best on 4/4 folds: 0.97101 ± 0.00004; τ = 0.01 costs 0.000001 |
| Non-Latin skeleton blocker | about +0.00005 estimated; rejected |
| Stage-2 LightGBM tuning (lr 0.03, 127/511 leaves, min leaf 300, feature fraction 0.6, L2 5), 400k S1 | all within ±0.0001 of the production parameters; current parameters kept |
| LightGBM + CatBoost blend (0.7 / 0.3) | +0.00011 at 400k S1, but **−0.00013 at full 1.2M S1**; rejected |
| LightGBM L2 = 5 at full size | 0.972177 vs 0.97217 (tie); kept L2 = 1 |
| **Cross-encoder fusion on borderline pairs** (0.02 < p < 0.995; 503k fold-0 pairs, 2.86M test pairs) | **+0.0104** on held-out half B (0.97179 → 0.98218); singleton F0.5 0.968 → 0.980; borderline AUC 0.916 → 0.949 (cross-encoder alone) |

These components were planned but **not run** because of compute and time: a neural cross-encoder, dense embeddings, a fine-tuned bi-encoder, and synthetic augmentation. See section 12.

## 6. Hard-negative analysis

The candidate set itself is the hard-negative pool. Every negative a model sees was retrieved by at least one TF-IDF blocker and survived pruning, so negatives are:

- same or similar names in other locations
- same address, different business
- generic chain names
- near-duplicate sibling entities

Stage 2 adds competition features that explicitly contrast each pair with the other S1s that want the same target. Remaining false positives on fold 0 (`reports/error_analysis.csv`):

| FP category | Pairs | Mean p |
|---|---|---|
| Same name and street, **conflicting building number** | 2,504 | 0.83 |
| Near-duplicate record (likely a sibling entity) | 2,387 | 0.89 |
| Weak evidence in both fields | 2,351 | 0.86 |
| Generic name collision | 1,237 | 0.88 |
| Same address, different business | 641 | 0.86 |
| Same name, target address missing | 608 | 0.86 |

## 7. Singleton analysis

- **Final singleton F0.5 is 0.9689.** 3.1% of true singletons receive a false match.
- **Singleton false matches cost 0.0017 of macro F0.5** (768 S1), out of a total loss of 0.0280.
- **How singletons are protected:**
  - the per-S1 expected-F rule compares predicting nothing (Π(1−p)) against every top-k prefix
  - a minimum top-candidate probability of 0.6
  - exclusivity
- **Why min_p = 0.6 over 0.5:** they tie on overall F0.5 (0.97205 vs 0.97207), but 0.6 raises singleton F0.5 from 0.959 to 0.969. That is the safer choice given the test set's higher target density.

## 8. Source-specific analysis

- **Candidate recall** is essentially identical for S2 and S3: 0.98159 vs 0.98165.
- **One shared model with a source indicator** is used (`src` feature). Separate S2/S3 models were not trained. The S2 and S3 noise differs in style (S2: uppercase addresses and more Indic-script names; S3: full state names, more DBA constructions), but string features normalize most of it and recall was balanced.
- **Cross-source evidence** enters through the sibling features: similarity of a target to the S1's other plausible candidates, including same-source maxima.

## 9. Final architecture

```
S1, S2, S3 TSVs
  → normalization v3 (names: canonical legal forms, core, sorted, phonetic/transliteration skeleton,
                      domain word-segmentation; addresses: abbreviations incl. France-scoped,
                      regions, numbers, postal)
  → blocking (country-scoped TF-IDF, word level):
        reverse name+addr top-5 | reverse addr top-3 | reverse name top-3
        forward name+addr top-30 | forward addr top-10
  → union (about 60/S1) → LightGBM pruner on retrieval features → top-12/S1
  → 51 pair features + retrieval features + name-frequency features
  → stage-1 LightGBM pair scorer, 5-fold cross-fitted (OOF p1)
  → cascade filter p1 >= 0.01 (about 4.65/S1)  == candidate_pairs.tsv
  → context within survivors (per-S1 rank/gap/set stats, per-target competition margin) + sibling consistency
  → stage-2 LightGBM
  → cross-encoder (multilingual-e5-small) on borderline pairs (0.02 < p < 0.995), logistic fusion with p
  → exclusivity (each target keeps its best S1) → per-S1 expected-F0.5 subset (min_p 0.6)
  → matching_results.tsv
```

## 10. Final configuration

- **Normalization** `NORM_VERSION = "v3"`.
- **Blockers** as in section 9. Word TF-IDF with `max_df = 0.02` and sublinear TF, fit per split on target text only.
- **Pruner:** LightGBM, lr 0.1, 63 leaves, 200 rounds, trained on 150k S1 from folds 1–4; top_n = 12.
- **Stage 1:** LightGBM binary, lr 0.05, 255 leaves, min_data_in_leaf 100, feature/bagging fraction 0.8, L2 1.0, early stopping on 10% of S1 groups; 600k S1 per fold model; 71 features.
- **Stage 2:** same parameters. It uses the 71 stage-1 features plus 13 context and 6 sibling features. It is trained on 1.2M S1; easy negatives (p1 < 0.005) are kept at 25% with weight 4.
- **Cascade:** candidate set = pairs with stage-1 p1 ≥ 0.01. Stage-2 context and sibling features are computed within the survivors only.
- **Decision:** exclusivity, then expected-F0.5 with `min_p = 0.6` and `extra_mass = 0.2`.
- **Test-time stage-1** is the average of the 5 fold models. Averaging was checked not to shift the score distribution: 4.4% vs 4.5% of pairs in the uncertain band.

## 11. Final test candidate statistics

| Statistic | Value |
|---|---|
| Test S1 | 1,732,544 (all present in both output files) |
| Candidate pairs | **8,877,239 (5.12 per S1, median 5, p95 = 9, max = 12)** |
| S1 with no candidates | 19,706 (the cascade found nothing plausible) |
| Candidates per S1 by country | US 5.06, India 4.99, France 5.71 |
| Reduction ratio vs same-country all-pairs | > 99.9999% |
| Predicted matches | 5,814,688 pairs (3.36 per S1) |
| S1 predicted with no match | 97,975 (5.7%; train singleton rate is 5.6%) |

Predictions by country:

| Country | Empty-prediction rate | Mean matches per S1 |
|---|---|---|
| US | 5.6% | 3.46 |
| India | 5.9% | 3.33 |
| France | 4.8% | 3.48 |

France (unseen in training) behaves like the training countries.

## 12. Known failure modes

Loss decomposition on fold 0 (share of 1 − macro F0.5 = 0.0280):

| S1-level error | S1 count | Loss | Of which unavoidable given candidates |
|---|---|---|---|
| Non-singleton missing some matches (no FP) | 69,290 | 0.0132 | 0.0043 |
| Non-singleton predicted empty | 3,104 | 0.0070 | 0.0015 |
| Non-singleton with a false match | 9,630 | 0.0060 | 0.0002 |
| Singleton false match | 768 | 0.0017 | 0 |

Main false-negative causes:

| Cause | Pairs |
|---|---|
| Blocker misses (41% of them non-Latin names, 33% missing target address) | 28k |
| Moderate evidence below the gate | 27.9k |
| Missing target address (name-only evidence, often generic) | 22.9k |
| Transliteration | 6.1k |
| Alias / replaced name | 4.1k |

Other limitations:

- **Stage-2 tuning has converged.** Parameter variants, CatBoost, and blends all landed within noise of the production model.
- **The cross-encoder is under-trained.** It saw only 80k borderline pairs (about 11 minutes on Apple MPS) because of the deadline, and one model scores both validation and test. More training data, cross-fitted models feeding stage 2 directly, and a wider band are the clearest next gains.
- **Engineering note:** padding every batch to a fixed length (96 tokens) was essential on MPS. Variable-length batches made the GPU cache a compiled graph per shape until the machine swapped.
- **Not attempted because of compute:**
  - **The non-Latin skeleton reverse blocker** (`rev_skel_nonascii`) was run on train but not integrated, because that requires a full re-train.
- **Test density risk.** The test set has about 23% more targets per S1 than train, likely more distractors. Precision on test may be slightly below validation. The decision gate was chosen conservatively for this reason.
- **French-specific noise** has no training signal. Its handling relies on language-agnostic features and generic plus France-scoped abbreviation rules.

## 13. Reproduction instructions

From `student_resource/`, with the dataset TSVs in `dataset/train` and `dataset/test`:

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r code/business_entity_resolution/requirements.txt
brew install libomp          # macOS only, required by LightGBM
bash code/business_entity_resolution/run_all.sh
```

This regenerates normalization caches, blockers, candidates, features and models, writes `output/candidate_pairs.tsv` and `output/matching_results.tsv`, and runs the official validator.

- **Runtime** on an Apple M5 (10 cores, 16 GB) is about 9–10 hours end to end, dominated by the TF-IDF blockers (about 2.5 h train, 2 h test).
- **Caching:** every stage is cached under `cache/`, keyed by configuration, so a rerun skips finished stages.
- **Peak memory** is about 7 GB.
