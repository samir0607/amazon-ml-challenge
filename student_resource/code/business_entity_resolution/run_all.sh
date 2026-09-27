#!/usr/bin/env bash
# One command, run from student_resource/:  bash code/business_entity_resolution/run_all.sh
# data (dataset/{train,test}) -> normalization -> blocking -> features -> models -> output/ -> validator
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
export PYTHONPATH=code/business_entity_resolution/src
NAME=${NAME:-v2}; FEAT=${FEAT:-v1}; TOPN=${TOPN:-12}
BLOCKERS="rev_nameaddr rev_addr rev_name fwd_nameaddr fwd_addr"
$PY -u code/business_entity_resolution/scripts/run_blockers.py train $BLOCKERS
$PY -u code/business_entity_resolution/scripts/train_matcher.py --tag "$NAME" --feat-tag "$FEAT" --top-n "$TOPN" --train-s1 600000 --final
$PY -u code/business_entity_resolution/scripts/run_blockers.py test $BLOCKERS
$PY -u code/business_entity_resolution/scripts/predict_test.py --name "${NAME}_top${TOPN}"
# cross-encoder (multilingual-e5-small, MIT) on borderline pairs + logistic fusion, then final matches
$PY -u code/business_entity_resolution/scripts/ce_lite.py --val "val_pred_${NAME}_top${TOPN}.parquet" --val-col p --n-train 80000
$PY -u code/business_entity_resolution/scripts/ce_extend_band.py --val "val_pred_${NAME}_top${TOPN}.parquet" --hi2 0.995
# round 2: continue training the cross-encoder on 100k more borderline pairs, fuse both
$PY -u code/business_entity_resolution/scripts/ce_round2.py --stage val --val "val_pred_${NAME}_top${TOPN}.parquet"
$PY -u code/business_entity_resolution/scripts/ce_round2.py --stage test --val "val_pred_${NAME}_top${TOPN}.parquet"
# conservative prior shift chosen on the public leaderboard (test has more distractors per S1)
PRIOR_ODDS=${PRIOR_ODDS:-0.35}
$PY -u code/business_entity_resolution/scripts/finalize_ce.py --mode full --prior-odds "$PRIOR_ODDS"
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
