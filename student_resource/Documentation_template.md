# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Transformers
**Team Members:** Yash Bindal, Samir Gupta, Shreyas Jain, Nikunj Sharma
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We use a blocking-plus-two-stage-GBDT pipeline built entirely on the provided data.

- **Key innovation:** *reverse* blocking. Every S2/S3 record retrieves its best S1 entities, which exploits the fact that each record belongs to at most one S1. Its top-1 alone captures 94.6% of true pairs at 4.7 candidates per S1.
- **Precision:** a stage-2 matcher uses competition features (how strongly each S1 beats the other S1s competing for the same record) and sibling-consistency features. A per-entity expected-F0.5 decision rule treats "no match" as a first-class outcome.
- **Candidates:** a learned-blocking cascade (retrieval, then pruner, then pair scorer) keeps only **4.65 candidates per S1 on validation (5.12 on test)** at **98.0% candidate recall**.
- **Neural re-scoring:** a multilingual cross-encoder (intfloat/multilingual-e5-small, MIT, 118M parameters) re-scores the borderline pairs.
- **Result:** held-out macro F0.5 of **≈0.982** (0.98218 on a held-out half of the validation fold, vs 0.97179 without the cross-encoder).

---

## 2. Methodology

### 2.1 Problem Analysis

Measured on the training data:

- **Structure:**
  - 2.21M S1, 5.03M S2 and 5.29M S3 records
  - 5.6% of S1 are singletons; the mean is 3.46 matches per S1 (maximum 11)
  - every S2/S3 record matches at most one S1, and 26% match none
  - the country label agrees on 100% of true pairs
  - test adds France (15% of test S1), unseen in training
- **Name noise:**
  - word shuffles and doubled words
  - character and leet typos (`Denta1`, `Si1ver`)
  - diacritics and legal-suffix swaps (Pvt/Private, Ltd/Limited, L.L.C.)
  - prefixes/suffixes such as `--`, `Sri` and `Center`
  - DBA constructions (`X trading as Y`, `formerly:`)
  - domain-style names (`magnatraders.com`)
  - randomly replaced names
  - Indic-script names (15% of S2 names): Devanagari, Tamil, Kannada, Telugu, Malayalam, Bengali, Gujarati
- **Address noise:**
  - uppercase and abbreviations (RD/ST/CT)
  - component reordering
  - state code vs full name vs native script
  - injected wrong city
  - partial addresses; 3.3% of S2/S3 addresses are missing
- **Generic names.** Many distinct businesses share generic names, so name-only retrieval reaches only 68% recall even at 50 candidates. The address is essential for disambiguation.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage GBDT classifier with graph-style context features (hybrid)

**Core Innovation:** Our three main contributions:

1. **Reverse (target → S1) TF-IDF retrieval** combined with forward retrieval and a learned pruner.
2. **Exclusivity-aware competition features and sibling-consistency features** from out-of-fold stage-1 scores.
3. **A per-S1 expected-F0.5 decision rule.** It explicitly scores the empty prediction, which protects singletons.

---

## 3. Candidate Generation (Blocking)

**Normalization (deterministic; raw text is kept alongside):**
- **Names:**
  - NFKD/ASCII folding
  - legal-form canonicalization, including transliterated forms matched by phonetic skeleton (`praaivett` → private)
  - core name with legal forms and fillers removed
  - sorted token set
  - consonant skeleton that bridges transliteration and typos
  - DBA/alias split
  - dictionary word-segmentation of domain names, with a vocabulary built only from S1 names
- **Addresses:**
  - abbreviation expansion (generic, plus rules scoped to the France country label)
  - US/India region canonicalization, including native-script state names via skeleton
  - number, postal-like and region tokens

**Blocking keys used:** country-scoped sparse TF-IDF over word tokens with IDF weighting.

| Blocker | Direction | Text | Top-k |
|---|---|---|---|
| Reverse name+address | target → S1 | core name tokens + skeleton tokens + address tokens | 5 |
| Reverse address | target → S1 | address tokens | 3 |
| Reverse name | target → S1 | name + skeleton tokens | 3 |
| Forward name+address | S1 → target | core name tokens + skeleton tokens + address tokens | 30 |
| Forward address | S1 → target | address tokens | 10 |

- **Pruning:** the union (about 60 candidates per S1, 98.56% recall) is pruned in two learned steps. The first is a LightGBM ranker on retrieval features. Those features are:
  - per-blocker scores and ranks
  - number of blockers that retrieved the pair
  - the target's global best and second-best S1 score
  - gap to the S1's best candidate
- **Step one keeps the top-12 per S1** (98.19% recall).
- **Step two is a learned-blocking cascade.** A cross-fitted LightGBM pair scorer over string/numeric features keeps only pairs with score ≥ 0.01. The survivors are exactly `candidate_pairs.tsv`, and the final matcher runs inference only on them.
- **Effect:** 12 → **4.65 candidates per S1** on validation with no F0.5 loss (0.97165 → 0.97173 at equal training size). A retrieval-only threshold at a similar size would cost 0.004 F0.5.
- **Rejected blockers:** exact-key blockers were measured and dropped (at most +0.0001 recall). Char-n-gram retrieval over all queries was too slow for too little gain.

**Candidate pairs generated:**
- test: **8,877,239** for 1,732,544 S1: mean **5.12**, median 5, p95 9, max 12; reduction ratio > 99.9999%
- validation: 4.65 per S1

**How we ensured true matches were not lost:**
- Every blocker and every union/pruning depth was measured for pair recall, S1 full coverage, mean/p95/max candidates and per-source recall.
- Leave-one-out ablations confirmed each kept blocker's contribution.
- The pruning depth was chosen at the knee of the curve: top-10 gives 98.02%, **top-12 gives 98.19%**, and top-20 gives 98.44%.
- The cascade threshold was chosen by retraining the final matcher at τ ∈ {0, 0.01, 0.03, 0.05} and cross-validating over 5 folds. τ = 0.01 keeps 99.8% of the top-12 recall.
- Final fold-0 candidate recall is **97.97%**.
- Rejected additions: a 6th, non-Latin-name blocker (+0.05% recall, about +0.00005 F0.5) and exact-key blockers (+0.0001 recall).

---

## 4. Matching Model

**Features used:**
- **Name features:**
  - equality of core, sorted-token and skeleton forms
  - RapidFuzz ratio, token-sort, token-set and partial ratio
  - Jaro-Winkler; Levenshtein on the space-free core
  - skeleton token-set/ratio (transliteration)
  - token Jaccard and skeleton Jaccard
  - IDF-weighted overlap (min-normalized, union-normalized, max shared IDF, IDF mass per side)
  - alias token-set match; domain partial match
  - length ratio and token-count difference
  - non-Latin-script flag
  - generic-name frequency (targets and S1 sharing the sorted core name)
- **Address features:**
  - ratio, token-set and partial ratio
  - token Jaccard; IDF-weighted overlap and rarest shared token
  - number Jaccard, shared count, first-number (house number) equality, number conflict
  - postal-code equality/conflict and region equality/conflict
  - missingness per side, length ratio, non-Latin flag
- **Other:**
  - cross features: min, max and product of name and address similarity
  - retrieval features: per-blocker scores and ranks, blocker count, pruner probability and rank
  - source indicator
  - **stage-2 competition context** from out-of-fold stage-1 probabilities:
    - rank and gap within the S1's candidates
    - rank, gap and margin against the best *other* S1 for the same target (exclusivity)
    - set-level expected match count and number of candidates above 0.5 / 0.2
  - **sibling consistency:** similarity of the target to the S1's other plausible records (S2↔S3 and within-source), weighted by their probability

**Model type:** Two-stage LightGBM (binary log-loss) plus a cross-encoder re-scorer.
- **Stage 1:** 5-fold cross-fitted, 600k S1 per fold model, 71 features.
- **Stage 2:** 90 features, trained on 1.2M S1 with 25% sampling of easy negatives and importance weights.
- **Tuning:** screened on a fixed 400k-S1 sample, then confirmed at full size:
  - learning rate, leaves, min leaf size, feature fraction and L2 all landed within ±0.0001 of the production parameters
  - a LightGBM + CatBoost blend gained +0.00011 at 400k S1 but lost 0.00013 at full size, so it was rejected
- **Negatives:** hard negatives come from the candidate set itself (retrieved lookalikes: same brand in another location, same address, generic names, sibling entities).
- **Test-time stage 1:** the average of the 5 fold models.

- **Cross-encoder:**
  - Fine-tuned `intfloat/multilingual-e5-small` (MIT, 118M parameters) on 80k borderline training pairs (stage-1 p between 0.02 and 0.98, folds 1–4). Input is the raw text `name | address | country` for both records, max 96 tokens.
  - At inference it scores only borderline candidate pairs (stage-2 p between 0.02 and 0.995): 2.86M of the 8.88M test candidates.
  - Its logit is fused with the stage-2 logit by logistic regression, fitted on half of the validation fold.
  - It reads Indic scripts and transliterations directly: borderline AUC rises from 0.916 to 0.949.

**Threshold selection method:**
1. **Exclusivity:** each target keeps only its best S1.
2. **Per-S1 expected-F0.5 subset selection:** candidates are sorted by probability, and the prefix k maximizing plug-in expected F0.5 is chosen. The empty set is scored by Π(1−p).
3. **Minimum top-candidate probability of 0.6**, with an expected 0.2 matches missed by blocking.

These parameters were tuned on half of the validation fold and confirmed on the other half. The rule beat fixed thresholds from 0.3 to 0.8.

---

## 5. Results & Error Analysis

**F_0.5 Score (macro):** **0.98218** with the cross-encoder, on a held-out half of the validation fold (about 220k S1 unseen by the cross-encoder and fusion; 0.97179 without it). The GBDT-only pipeline scores 0.97217 on the full validation fold of 441,365 S1, and 5-fold cross-validation of its decision rule gives 0.97101 ± 0.00004.

| Metric | Value |
|---|---|
| Micro precision | 0.9926 |
| Micro recall | 0.9420 |
| Singleton F0.5 | 0.969 |

Progression:

| Model | Macro F0.5 |
|---|---|
| Stage-1 only | 0.9649 |
| + competition context | 0.9669 |
| + set features | 0.9685 |
| + sibling features | 0.9702 |
| + more data | 0.9721 |
| + learned-blocking cascade (4.65 candidates per S1) | 0.9722 |
| + cross-encoder fusion on borderline pairs (final) | **0.9820** |

**Common false positives (wrong merges):**
- same name and street with a conflicting building number
- near-duplicate sibling entities
- generic-name collisions
- same address, different business
- name-only matches where the target address is missing

Singleton false matches affect 3.1% of true singletons.

**Common false negatives (missed matches):**
- about 28k true pairs are never retrieved; 41% of those have non-Latin names and 33% missing addresses
- moderate-evidence pairs below the precision gate
- missing target address with a generic name (inherently ambiguous)
- transliteration and alias/replaced names

---

## 6. Conclusion

Treating exclusivity as signal made the biggest difference: in reverse blocking, in the per-target competition features, and in the decision rule. Together they delivered high precision (0.993 micro) at 12 candidates per S1. Macro F0.5 rewards getting each entity's whole set right, so modelling "no match" explicitly and adding set-level and sibling context mattered more than extra string features.

---

## Appendix

### A. Code Artefacts

Located in `code/business_entity_resolution/`:

- `src/ber/`: library modules (see `README.md`)
- `src/scripts/`: entry points
- `requirements.txt`: pinned versions
- `run_all.sh`: one command that regenerates both output files and runs the validator

The pipeline stages are:

1. `run_blockers.py` for train and test
2. `train_matcher.py --final`
3. `predict_test.py`

Every stage is cached. The pipeline makes no external calls. LightGBM (MIT), rapidfuzz (MIT) and scikit-learn (BSD) are the modelling libraries; the only pretrained model is intfloat/multilingual-e5-small (MIT, 118M parameters), fine-tuned locally on the training data.

### B. Additional Results

- `reports/final_report.md`: full blocking and matcher tables, ablations, singleton/source analysis, loss decomposition
- `reports/candidate_analysis.csv`, `reports/ablation_results.csv`: all logged experiments
- `reports/error_analysis.csv` and `reports/error_analysis.html`: categorized FP/FN with examples
- `experiments/experiment_log.csv`: every run


## Leaderboard calibration (final)

Validation overstates test performance, because test has about 23% more S2/S3 records per S1 (more distractors). Public-leaderboard results:

| Submission | Validation (half B) | Leaderboard |
|---|---|---|
| Cross-encoder can only add matches | 0.978 | 0.950 |
| Round-1 fusion, prior odds ×1 | 0.982 | 0.971 |
| Round-1 fusion, prior odds ×0.5 | — | 0.972 |
| Round-1 fusion, prior odds ×0.25 | — | 0.973 |
| Round-2 fusion, prior odds ×0.05 | — | 0.971 |
| Round-2 fusion, prior odds ×0.1 | — | 0.973 |
| Round-2 fusion, prior odds ×0.15 | — | lower than ×0.25 |
| Round-2 fusion, prior odds ×0.25 | 0.984 (at ×1) | 0.975 |
| **Round-2 fusion, prior odds ×0.35 (final)** | — | **best, slightly above 0.975** |

Each fused match probability's odds are multiplied by 0.35 (`finalize_ce.py --prior-odds 0.35`), so a pair needs about 3× more evidence before it counts as a match. The value was chosen by a small scan of k on the public leaderboard; the peak is broad, spanning roughly 0.25 to 0.35.

- **Why it works:** it corrects for the lower match rate among borderline test pairs.
- **Evidence:** adding matches consistently hurt on the leaderboard and removing them helped, the opposite of validation.
- **Final test predictions:** 5,731,144 matched pairs.
