#!/bin/bash -l
#$ -l h_rt=30:00:00
#$ -N embed-metadata
#$ -j y
#$ -o embed-metadata.log
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

python -m ingestion.ingest --metadata-file data/metadata/metadata.jsonl --skip-fulltext