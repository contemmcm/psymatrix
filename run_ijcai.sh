#!/bin/bash

python -m psymatrix.finetune -e "ijcai25" --train-split-usage 100 --test-split-usage 100
python -m psymatrix.finetune -e "ijcai25" --train-split-usage 80 --test-split-usage 80
python -m psymatrix.finetune -e "ijcai25" --train-split-usage 60 --test-split-usage 60
python -m psymatrix.finetune -e "ijcai25" --train-split-usage 40 --test-split-usage 40
python -m psymatrix.finetune -e "ijcai25" --train-split-usage 20 --test-split-usage 20

