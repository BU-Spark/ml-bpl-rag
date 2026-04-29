#!/bin/bash -l
#$ -P sparkgrp
#$ -l h_rt=12:00:00
#$ -N bpl-graph-extract
#$ -j y
#$ -o logs/graph-extract.log
#$ -pe omp 2

source /etc/profile.d/modules.sh
module load miniconda
conda activate spark-rag

cd /projectnb/sparkgrp/ml-bpl-rag-data-subset/temp/ml-bpl-rag/current_spring2026

mkdir -p data/graph logs

echo "Starting entity extraction at $(date)"
echo "CPUs available: $NSLOTS"

python -m graph.extract_entities --all --concurrency 50

echo "Entity extraction complete at $(date)"
