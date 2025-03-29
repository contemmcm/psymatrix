#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m psymatrix.finetune -e "icml25" -m "openai-community/gpt2" --train-split-usage 100 --test-split-usage 100 > output_icml_100.log 2>&1 &
