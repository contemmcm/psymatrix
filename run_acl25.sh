#!/bin/bash
MODEL="distilbert/distilbert-base-uncased"
DATASET="PsyMatrix/cls_20newsgroups_SubjectTextVsLabel__BaseDefault"

DATASET_SIZE=(10 20 30 40 50 60 70 80 90 100)
LABEL_NOISE=10

# 10% label noise
for p in "${DATASET_SIZE[@]}"; do
    python -m psymatrix.finetune -e "acl25" -m "$MODEL" -d "$DATASET" \
        --train-split-usage "$p" --test-split-usage "$p" \
        --train-label-noise "$LABEL_NOISE" --test-label-noise "$LABEL_NOISE"
done
