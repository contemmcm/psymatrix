#!/bin/bash

DATASET=$1
TRAIN_SPLIT=$2
TEST_SPLIT=$3

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
python scripts/dump_documents.py --name_or_path "$DATASET" --split "$TRAIN_SPLIT"

#
# Extract features
#
python -m scripts.feat_lang --name_or_path "$DATASET" --split "$TRAIN_SPLIT"
python scripts/feat_topic_lda.py --name_or_path "$DATASET" --split "$TRAIN_SPLIT"
python scripts/feat_psy_textstat.py --name_or_path "$DATASET" --split "$TRAIN_SPLIT"
python scripts/feat_psy_taaco.py --name_or_path "$DATASET" --split "$TRAIN_SPLIT"

if [[ "$IMAGE_EXISTS" -eq 1 ]]; then
    docker run -it --rm -v `pwd`/data/:/root/data --user $(id -u):$(id -g) \
    cohmetrix python3 scripts/feat_psy_cohmetrix.py --name_or_path "$DATASET" --split "$TRAIN_SPLIT" --processes "$CPU_COUNT"
else
    python scripts/feat_psy_cohmetrix.py --name_or_path "$DATASET" --split "$TRAIN_SPLIT" --processes "$CPU_COUNT"
fi


#
# Extract meta-features
#
python scripts/feat_mfe.py --name_or_path "$DATASET" --split "$TRAIN_SPLIT"

python -m scripts.metafeatures --name_or_path "$DATASET" --split "$TRAIN_SPLIT" --feature lang
python -m scripts.metafeatures --name_or_path "$DATASET" --split "$TRAIN_SPLIT" --feature topic_lda
python -m scripts.metafeatures --name_or_path "$DATASET" --split "$TRAIN_SPLIT" --feature textstat
python -m scripts.metafeatures --name_or_path "$DATASET" --split "$TRAIN_SPLIT" --feature taaco
python -m scripts.metafeatures --name_or_path "$DATASET" --split "$TRAIN_SPLIT" --feature cohmetrix
python -m scripts.metafeatures --name_or_path "$DATASET" --split "$TRAIN_SPLIT" --feature cohmetrix
