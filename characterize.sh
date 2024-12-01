#!/bin/bash

DATASET=$1
TRAIN_SPLIT=$2
TEST_SPLIT=$3
COHMETRIX_BATCH_SIZE=50

check_docker_cohmetrix() {
  IMAGE_NAME="cohmetrix"

  if docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    return 1
  else
    return 0
  fi
}

CPU_COUNT=$(nproc)

check_docker_cohmetrix
IS_COHMETRIX_DOCKER_AVAILABLE=$?  # Capture the return value

#
# Download the dataset
#
python3 scripts/dump_documents.py --name_or_path "$DATASET" --split "$TRAIN_SPLIT"

#
# Extract features
#
python3 -m scripts.feat_lang --name_or_path "$DATASET" --split "$TRAIN_SPLIT"
python3 scripts/feat_topic_lda.py --name_or_path "$DATASET" --split "$TRAIN_SPLIT"
python3 scripts/feat_psy_textstat.py --name_or_path "$DATASET" --split "$TRAIN_SPLIT"
python3 scripts/feat_psy_taaco.py --name_or_path "$DATASET" --split "$TRAIN_SPLIT"

# if [[ "$IMAGE_EXISTS" -eq 1 ]]; then
#   docker run -it --rm -v `pwd`/data/:/home/ubuntu/data \
#     cohmetrix python3 scripts/feat_psy_cohmetrix.py --name_or_path "$DATASET" --split "$TRAIN_SPLIT" --processes "$CPU_COUNT" --batch-size $COHMETRIX_BATCH_SIZE
# else
#   python3 scripts/feat_psy_cohmetrix.py --name_or_path "$DATASET" --split "$TRAIN_SPLIT" --processes "$CPU_COUNT" --batch-size $COHMETRIX_BATCH_SIZE
# fi

python3 scripts/feat_psy_cohmetrix.py --name_or_path "$DATASET" --split "$TRAIN_SPLIT" --processes "$CPU_COUNT" --batch-size $COHMETRIX_BATCH_SIZE

#
# Extract meta-features
#
python3 scripts/feat_mfe.py --name_or_path "$DATASET" --split "$TRAIN_SPLIT"

python3 -m scripts.metafeatures --name_or_path "$DATASET" --split "$TRAIN_SPLIT" --feature lang
python3 -m scripts.metafeatures --name_or_path "$DATASET" --split "$TRAIN_SPLIT" --feature topic_lda
python3 -m scripts.metafeatures --name_or_path "$DATASET" --split "$TRAIN_SPLIT" --feature textstat
python3 -m scripts.metafeatures --name_or_path "$DATASET" --split "$TRAIN_SPLIT" --feature taaco
python3 -m scripts.metafeatures --name_or_path "$DATASET" --split "$TRAIN_SPLIT" --feature cohmetrix
