#!/bin/bash

#SBATCH -t 0-96:00:00
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH --output=test_%j_out.txt
#SBATCH --error=test_%j_err.txt

# To execute this, run one of the following commands: 
# sbatch -J FT01 --partition=informatik-mind --gpus=V100:1 run_ijcai.sbatch "openai-community/gpt2"

module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh

conda activate psymatrix

python -m psymatrix.finetune -e "icml25" -m "$1" --train-split-usage 5 --test-split-usage 5

conda deactivate
