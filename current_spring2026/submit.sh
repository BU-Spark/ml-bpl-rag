#!/bin/bash -l
#$ -l h_rt=30:00:00
#$ -N embed-fulltext
#$ -j y
#$ -o embed-fulltext.log
#$ -l gpus=1
#$ -l gpu_memory=16G
#$ -l gpu_c=7.5
#$ -pe omp 8

# =============================================================================
# Load config (universal params + task list)
# =============================================================================
source /etc/profile.d/modules.sh
module load miniconda
conda activate spark-rag

python -m ingestion.ingest --fulltext-dir data/fulltext/boston-traveler --skip-metadata