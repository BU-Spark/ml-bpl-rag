#!/bin/bash -l
#$ -l h_rt=5:00:00
#$ -N update-metadata
#$ -j y
#$ -o update-metadata.log
#$ -l gpus=1
#$ -l gpu_memory=16G
#$ -l gpu_c=7.5
#$ -pe omp 8

# =============================================================================
# Load config (universal params + task list)
# =============================================================================
# Load miniconda correctly for this SCC setup
module load miniconda

conda activate spark-rag

# Change to project root — critical for module imports
cd /projectnb/sparkgrp/ml-bpl-rag-data-subset/ml-bpl-rag-spring-2026/ml-bpl-rag/current_spring2026

echo "Starting at $(date)"

python scripts/update_abstracts_and_embeddings.py

echo "Done at $(date)"