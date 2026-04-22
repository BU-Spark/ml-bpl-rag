#!/bin/bash -l
#$ -P ivc-ml
#$ -l h_rt=24:00:00
#$ -N bpl-graph-embed-write
#$ -j y
#$ -o logs/graph-metadata-write.log
#$ -pe omp 1

source /etc/profile.d/modules.sh
module load miniconda
conda activate spark-rag

cd /projectnb/sparkgrp/ml-bpl-rag-data-subset/temp/ml-bpl-rag/current_spring2026

echo "Starting embed + write at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# 3. Embed GPT-4o-mini entities
# python -m graph.embed_entities --suffix metadata
python -m graph.write_graph --suffix metadata

echo "Embed + write complete at $(date)"
