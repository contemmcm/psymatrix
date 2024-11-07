#!/bin/bash
EXPERIMENT=$1
TARGET_DATASET=$2
TRAIN_SPLIT=$3

python -m psymatrix.psymatrix --experiment=$EXPERIMENT --target-dataset=$TARGET_DATASET --train-split=$TRAIN_SPLIT